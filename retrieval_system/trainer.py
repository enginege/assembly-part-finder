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
from .retrieval import RetrievalSystem
import gc

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
    def __init__(self, model, device, learning_rate=1e-4, part_batch_size=16, accumulation_steps=4):
        self.model = model
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        self.triplet_loss = TripletLoss().to(device)
        self.contrastive_loss = ContrastiveLoss().to(device)
        self.part_batch_size = part_batch_size
        self.early_stopping = EarlyStopping(patience=5)
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
                logging.warning(f"\nWarning: Graph {idx} in batch is None!")
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

    def train_epoch(self, dataloader, scaler, epoch):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad(set_to_none=True)

        # Get dataset's cache manager if it exists
        cache_manager = getattr(dataloader.dataset, 'cache_manager', None)

        # Add memory cleanup at the start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=True)):
            try:
                # Let cache manager handle its own memory management
                if cache_manager:
                    cache_manager.clear_unused_cache()

                # Move data to device explicitly and clear unnecessary tensors
                if 'cached_embeddings' in batch:
                    with torch.cuda.amp.autocast():
                        loss = self.compute_loss_from_cache(batch)
                else:
                    # Process batch in smaller chunks
                    with torch.cuda.amp.autocast():
                        assembly_img = batch['assembly_image'].to(self.device)
                        loss = self.model.compute_loss(
                            assembly_img,
                            batch['part_images'].to(self.device),
                            [g.to(self.device) for g in batch['graph']]
                        )
                        del assembly_img  # Explicitly delete unnecessary tensors

                # Scale loss and backward pass
                loss = loss / self.accumulation_steps
                scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    # Clear cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                total_loss += loss.item() * self.accumulation_steps

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"OOM error in batch {batch_idx}. Skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
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

        # Get dataset's cache manager if it exists
        cache_manager = getattr(val_loader.dataset, 'cache_manager', None)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                try:
                    # Let cache manager handle its own memory management
                    if cache_manager:
                        cache_manager.clear_unused_cache()

                    # Move data to device
                    assembly_images = batch['assembly_image'].to(self.device)
                    part_images = batch['part_images'].to(self.device)
                    graphs = [g.to(self.device) for g in batch['graph']]

                    # Compute loss
                    loss = self.model.compute_loss(assembly_images, part_images, graphs)
                    total_val_loss += loss.item()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.error(f"\nOOM error during validation. Skipping batch...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        return total_val_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs):
        logging.info(f"\nStarting training for {epochs} epochs...")
        scaler = GradScaler()
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader, scaler, epoch)

            # Validation phase
            val_loss = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping check
            self.early_stopping(val_loss)

            logging.info(f"Epoch {epoch+1}/{epochs}")
            logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logging.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model with full configuration and its index
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join("saved_models", "best_model.pth")
                best_index_path = os.path.join("saved_models", "best_model_index.pkl")

                # Save model
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'embedding_dim': self.model.embedding_dim,
                    'model_config': {
                        'embedding_dim': self.model.embedding_dim
                    }
                }
                torch.save(checkpoint, best_model_path)

                # Build and save index for best model
                retrieval_system = RetrievalSystem(self.model, self.device)
                retrieval_system.build_index(train_loader)
                retrieval_system.save_index(best_index_path)

                logging.info(f"Saved best model and index with validation loss: {val_loss:.4f}")

            if self.early_stopping.early_stop:
                logging.info("Early stopping triggered")
                break