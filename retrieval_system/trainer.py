import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
from .losses import TripletLoss, ContrastiveLoss
from torch_geometric.data import Data

class ModelTrainer:
    def __init__(self, model, device, learning_rate=1e-4, part_batch_size=16):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.triplet_loss = TripletLoss().to(device)
        self.contrastive_loss = ContrastiveLoss().to(device)
        self.part_batch_size = part_batch_size

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

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        scaler = torch.cuda.amp.GradScaler()

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

    def train(self, train_loader, epochs):
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training with {len(train_loader.dataset)} assemblies")
        print("-" * 60)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = len(train_loader)

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move data to device
                    assembly_images = batch['assembly_image'].to(self.device)
                    part_images = batch['part_images'].to(self.device)
                    graphs = [g.to(self.device) for g in batch['graph']]

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    loss = self.model.compute_loss(assembly_images, part_images, graphs)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    # Update statistics
                    total_loss += loss.item()
                    avg_loss = total_loss / (batch_idx + 1)

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'batch': f'{batch_idx + 1}/{num_batches}'
                    })

                except Exception as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    print("Batch keys:", batch.keys())
                    continue