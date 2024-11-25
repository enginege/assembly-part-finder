import torch
import torch.nn as nn
from .feature_extractors import ImageFeatureExtractor, GraphFeatureExtractor
import torch.nn.functional as F

class RetrievalModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.assembly_encoder = ImageFeatureExtractor(output_dim=embedding_dim)
        self.part_encoder = ImageFeatureExtractor(output_dim=embedding_dim)
        self.graph_encoder = GraphFeatureExtractor(
            input_dim=1,  # From dataset's node features
            hidden_dim=embedding_dim*2,
            output_dim=embedding_dim
        )
        # Temperature parameter for InfoNCE loss
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def compute_loss(self, assembly_images, part_images, graphs):
        """Compute the combined loss for the model"""
        batch_size = len(assembly_images)

        # Ensure consistent dtype
        assembly_images = assembly_images.to(dtype=torch.float32)
        part_images = part_images.to(dtype=torch.float32)

        with torch.cuda.amp.autocast():
            # Get embeddings
            assembly_emb = F.normalize(self.encode_assembly(assembly_images), dim=1)

            # Process parts
            part_embs_list = []
            for parts in part_images:
                if len(parts) == 0:
                    # Handle empty parts with zero embedding
                    part_emb = torch.zeros(1, self.embedding_dim, device=parts.device)
                else:
                    part_emb = F.normalize(self.encode_part(parts), dim=1)
                part_embs_list.append(part_emb.mean(dim=0))  # Average part embeddings
            part_embs = torch.stack(part_embs_list)

            # Process graphs
            graph_embs_list = []
            for graph in graphs:
                # Ensure graph tensors are float32
                graph.x = graph.x.to(dtype=torch.float32)
                graph_emb = F.normalize(self.encode_graph(
                    graph.x,
                    graph.edge_index,
                    None
                ), dim=1)
                graph_embs_list.append(graph_emb.squeeze(0))
            graph_embs = torch.stack(graph_embs_list)

            # Compute InfoNCE losses
            assembly_part_logits = torch.matmul(assembly_emb, part_embs.T) / self.temperature
            part_graph_logits = torch.matmul(part_embs, graph_embs.T) / self.temperature

            # Labels are the diagonal indices (matching pairs)
            labels = torch.arange(batch_size, device=assembly_images.device)

            # Compute cross entropy losses
            loss_assembly_part = F.cross_entropy(assembly_part_logits, labels)
            loss_part_graph = F.cross_entropy(part_graph_logits, labels)

            # Total loss is the average of both losses
            total_loss = (loss_assembly_part + loss_part_graph) / 2

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