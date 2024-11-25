import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ImageFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        # Use ResNet18 instead of ResNet50 for lower memory usage
        self.backbone = models.resnet18(pretrained=True)

        # Freeze early layers to save memory
        for param in list(self.backbone.parameters())[:-3]:
            param.requires_grad = False

        # Replace final FC layer
        self.backbone.fc = nn.Linear(512, output_dim)  # ResNet18 has 512 features

    def forward(self, x):
        with torch.cuda.amp.autocast():  # Use mixed precision
            features = self.backbone(x)
        return features

class GraphFeatureExtractor(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        if batch is None:
            # If processing single graph, use mean pooling
            return x.mean(dim=0, keepdim=True)  # Shape: [1, output_dim]
        else:
            # If processing batch of graphs, use batch-wise mean pooling
            return global_mean_pool(x, batch)  # Shape: [batch_size, output_dim]