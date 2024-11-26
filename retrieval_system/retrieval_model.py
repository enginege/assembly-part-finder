import torch
import torch.nn as nn
from .feature_extractors import ImageFeatureExtractor, GraphFeatureExtractor
import torch.nn.functional as F
from torch_geometric.data import Data
from .losses import TripletLoss, ContrastiveLoss

class RetrievalModel(nn.Module):
    def __init__(self, embedding_dim=512, device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.assembly_encoder = ImageFeatureExtractor(output_dim=embedding_dim)
        self.part_encoder = ImageFeatureExtractor(output_dim=embedding_dim)
        self.graph_encoder = GraphFeatureExtractor(
            input_dim=1,  # From dataset's node features
            hidden_dim=embedding_dim*2,
            output_dim=embedding_dim
        )
        # Temperature parameter for InfoNCE loss
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        # Loss functions
        self.triplet_loss = TripletLoss().to(self.device)
        self.contrastive_loss = ContrastiveLoss().to(self.device)

    def compute_loss(self, assembly_images, part_images, graphs):
        """Compute the combined loss for the model"""
        batch_size = len(assembly_images)

        with torch.cuda.amp.autocast():
            # Get embeddings and normalize
            assembly_emb = F.normalize(self.encode_assembly(assembly_images), dim=1)  # [batch_size, embedding_dim]

            # Hard negative mining
            with torch.no_grad():
                similarities = torch.matmul(assembly_emb, assembly_emb.t())
                hardest_negative_idx = torch.argmax(similarities * (1 - torch.eye(batch_size, device=assembly_emb.device)), dim=1)

            # Process parts with weighted loss
            part_embs_list = []
            for parts in part_images:
                if len(parts) == 0:
                    part_emb = torch.zeros(1, self.embedding_dim, device=self.device)
                else:
                    part_emb = F.normalize(self.encode_part(parts), dim=1)
                part_embs_list.append(part_emb.mean(dim=0))
            part_embs = torch.stack(part_embs_list)  # [batch_size, embedding_dim]

            # Process graphs and normalize
            graph_embs = self.process_graphs_batch(graphs)  # [batch_size, embedding_dim]
            graph_embs = F.normalize(graph_embs, dim=1)

            # Compute losses with hard negatives
            loss_assembly_part = self.triplet_loss(
                assembly_emb,
                part_embs,
                part_embs[hardest_negative_idx]
            )

            # Generate labels for contrastive loss
            labels = torch.arange(batch_size, device=self.device)

            # Ensure all embeddings have the same dimensionality
            part_embs = part_embs.view(batch_size, -1)  # [batch_size, embedding_dim]
            graph_embs = graph_embs.view(batch_size, -1)  # [batch_size, embedding_dim]

            loss_part_graph = self.contrastive_loss(
                part_embs,
                graph_embs,
                labels
            )

            # Weight the losses
            total_loss = 0.7 * loss_assembly_part + 0.3 * loss_part_graph

        return total_loss

    def encode_assembly(self, x):
        return self.assembly_encoder(x)

    def encode_part(self, x):
        return self.part_encoder(x)

    def encode_graph(self, x, edge_index, batch=None):
        return self.graph_encoder(x, edge_index, batch)

    def forward(self, assembly_images, part_images, graphs):
        outputs = {}
        outputs['assembly_embeddings'] = self.encode_assembly(assembly_images)
        outputs['part_embeddings'] = self.encode_part(part_images)
        outputs['graph_embeddings'] = self.encode_graph(
            graphs.x, graphs.edge_index, graphs.batch if hasattr(graphs, 'batch') else None
        )
        return outputs

    def process_graphs_batch(self, graphs):
        """Process a batch of graphs"""
        graph_embeddings_list = []

        for graph in graphs:
            if graph is None:
                # Create a default graph with same embedding dimension
                node_features = torch.zeros(1, 2048, device=self.device)
                edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
                graph = Data(x=node_features, edge_index=edge_index)

            # Move graph to device
            graph = graph.to(self.device)

            # Get embedding
            with torch.cuda.amp.autocast():
                embedding = self.encode_graph(
                    graph.x,
                    graph.edge_index,
                    None
                )
                if len(embedding.shape) == 3:
                    embedding = embedding.squeeze(0)
                if len(embedding.shape) == 1:
                    embedding = embedding.unsqueeze(0)

            graph_embeddings_list.append(embedding)

        return torch.stack(graph_embeddings_list)