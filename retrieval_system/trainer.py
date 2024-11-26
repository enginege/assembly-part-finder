import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
from .losses import TripletLoss, ContrastiveLoss
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall
from torch.cuda.amp import autocast, GradScaler
import os

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelTrainer:
    def __init__(self, model, device, learning_rate=1e-4, part_batch_size=16):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        self.triplet_loss = TripletLoss().to(device)
        self.contrastive_loss = ContrastiveLoss().to(device)
        self.part_batch_size = part_batch_size
        self.early_stopping = EarlyStopping(patience=7)
        self.metrics = {
            'accuracy': Accuracy(task="binary").to(device),
            'precision': Precision(task="binary").to(device),
            'recall': Recall(task="binary").to(device)
        }

    def process_parts_in_batches(self, part_images):
        """Process part images in smaller batches to avoid OOM"""
        total_parts = part_images.size(0)
        embeddings_list = []

        for i in range(0, total_parts, self.part_batch_size):
            batch_end = min(i + self.part_batch_size, total_parts)
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    batch_embeddings = self.model.encode_part(part_images[i:batch_end])
                    embeddings_list.append(batch_embeddings.cpu())

        return torch.cat(embeddings_list, dim=0)

    def process_graphs(self, graphs):
        """Process a list of graphs"""
        graph_embeddings_list = []

        for idx, graph in enumerate(graphs):
            if graph is None:
                print(f"\nWarning: Graph {idx} in batch is None!")
                # Create a default graph
                graph = self._create_default_graph().to(self.device)

            # Move graph data to device
            graph_data = graph.to(self.device)

            # Process single graph
            with torch.cuda.amp.autocast():
                embedding = self.model.encode_graph(
                    graph_data.x,
                    graph_data.edge_index,
                    None  # No batch needed for single graph
                )
                if len(embedding.shape) == 3:
                    embedding = embedding.squeeze(0)
                if len(embedding.shape) == 1:
                    embedding = embedding.unsqueeze(0)

            graph_embeddings_list.append(embedding)

        return torch.stack(graph_embeddings_list)

    def _create_default_graph(self):
        """Create a default graph when one is missing"""
        node_features = torch.tensor(np.random.randn(1, 2048), dtype=torch.float)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return Data(x=node_features, edge_index=edge_index)

    def train_epoch(self, dataloader, scaler):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Print batch information
            print(f"\nProcessing batch {batch_idx}")
            print(f"Assembly IDs in batch: {batch['assembly_id']}")
            print(f"Number of graphs in batch: {len(batch['graph'])}")

            # Check graph paths
            if 'graph_path' in batch:
                print("Graph paths in batch:")
                for idx, path in enumerate(batch['graph_path']):
                    print(f"Assembly {batch['assembly_id'][idx]}: {path}")

            torch.cuda.empty_cache()

            try:
                self.optimizer.zero_grad(set_to_none=True)

                sub_batch_size = 2
                batch_size = len(batch['assembly_image'])
                batch_loss = 0

                for sub_idx in range(0, batch_size, sub_batch_size):
                    end_idx = min(sub_idx + sub_batch_size, batch_size)

                    # Get sub-batch data
                    assembly_img = batch['assembly_image'][sub_idx:end_idx].to(self.device)
                    sub_graphs = batch['graph'][sub_idx:end_idx]

                    with torch.cuda.amp.autocast():
                        assembly_embeddings = self.model.encode_assembly(assembly_img)
                        graph_embeddings = self.process_graphs(sub_graphs)

                        sub_loss = 0
                        for i in range(len(assembly_img)):
                            parts = batch['part_images'][sub_idx + i].to(self.device)
                            part_embeddings = self.process_parts_in_batches(parts)
                            part_embeddings = part_embeddings.to(self.device)

                            triplet_loss = self.triplet_loss(
                                assembly_embeddings[i].unsqueeze(0),
                                part_embeddings,
                                torch.roll(part_embeddings, 1, 0)
                            )

                            graph_emb = graph_embeddings[i].view(1, -1).expand(len(part_embeddings), -1)
                            contrastive_loss = self.contrastive_loss(
                                part_embeddings,
                                graph_emb,
                                torch.arange(len(part_embeddings)).to(self.device)
                            )

                            sub_loss += (triplet_loss + contrastive_loss) / len(assembly_img)

                    batch_loss += sub_loss.item()
                    scaler.scale(sub_loss).backward()

                    del assembly_embeddings, graph_embeddings, part_embeddings
                    torch.cuda.empty_cache()

                scaler.step(self.optimizer)
                scaler.update()

                total_loss += batch_loss

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in batch {batch_idx}. Skipping...")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        return total_loss / len(dataloader)

    def compute_metrics(self, embeddings, labels):
        """Compute accuracy, precision, and recall for embeddings"""
        similarities = torch.matmul(embeddings, embeddings.t())
        predictions = (similarities > 0.5).float()

        metrics = {}
        for name, metric in self.metrics.items():
            metrics[name] = metric(predictions, labels)
        return metrics

    def validate(self, val_loader):
        """Validate the model on the validation set"""
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                try:
                    # Move data to device
                    assembly_images = batch['assembly_image'].to(self.device)
                    part_images = batch['part_images'].to(self.device)
                    graphs = [g.to(self.device) for g in batch['graph']]

                    # Compute loss
                    loss = self.model.compute_loss(assembly_images, part_images, graphs)
                    total_val_loss += loss.item()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nOOM error during validation. Skipping batch...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        return total_val_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs):
        print(f"\nStarting training for {epochs} epochs...")
        scaler = GradScaler()
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader, scaler)

            # Validation phase
            val_loss = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping check
            self.early_stopping(val_loss)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join("saved_models", "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)

            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break