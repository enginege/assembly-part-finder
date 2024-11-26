import os
import torch
import networkx as nx
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch_geometric.data import Data

class AssemblyDataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_images=False):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None

        # Initialize lists to store paths and IDs
        self.assembly_ids = []  # Initialize first
        self.assembly_images = []
        self.part_images = []
        self.graph_paths = []

        # Find all assembly directories
        assembly_dirs = [d for d in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, d))]

        print("\nSearching for assemblies...")
        for assembly_dir in sorted(assembly_dirs, key=lambda x: int(x) if x.isdigit() else float('inf')):
            try:
                assembly_id = int(assembly_dir)
                assembly_path = os.path.join(root_dir, assembly_dir)

                # Find graph file
                graph_path = os.path.join(assembly_path, f'{assembly_id}_assembly.graphml')

                if os.path.exists(graph_path):
                    # Find images
                    images_dir = os.path.join(assembly_path, 'images')
                    if os.path.exists(images_dir):
                        assembly_img = os.path.join(images_dir, f'{assembly_id}_full_assembly.png')

                        # Get part images
                        part_imgs = []
                        for img_file in os.listdir(images_dir):
                            if img_file != f'{assembly_id}_full_assembly.png' and img_file.endswith('.png'):
                                part_imgs.append(os.path.join(images_dir, img_file))

                        if os.path.exists(assembly_img) and part_imgs:
                            self.assembly_ids.append(assembly_id)
                            self.assembly_images.append(assembly_img)
                            self.part_images.append(sorted(part_imgs))  # Sort part images for consistency
                            self.graph_paths.append(graph_path)

                            print(f"Added assembly {assembly_id}:")
                            print(f"  Graph path: {graph_path}")
                            print(f"  Assembly image: {assembly_img}")
                            print(f"  Number of parts: {len(part_imgs)}")

            except ValueError:
                continue

        print(f"\nDataset Loading Summary:")
        print(f"Total assemblies loaded: {len(self.assembly_ids)}")

    def load_graph(self, graph_path):
        """Load and process graph from graphml file"""
        try:
            print(f"\nAttempting to load graph from: {graph_path}")
            G = nx.read_graphml(graph_path)
            print(f"Loading graph from: {graph_path}")

            # Create a mapping of node names to integers
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, node_mapping)

            # Convert to PyTorch Geometric Data object
            edge_index = []
            for edge in G.edges():
                edge_index.append([edge[0], edge[1]])
                edge_index.append([edge[1], edge[0]])  # Add reverse edge for undirected graph

            num_nodes = G.number_of_nodes()

            # Handle empty graphs (no edges)
            if len(edge_index) == 0:
                # Create self-loops for each node
                edge_index = [[i, i] for i in range(num_nodes)]

            edge_index = torch.tensor(edge_index).t().contiguous()

            # Create node features
            x = torch.ones(num_nodes, 1, dtype=torch.float32)  # Explicitly set dtype

            data = Data(x=x, edge_index=edge_index.long())  # Ensure long dtype for indices
            print(f"Successfully loaded graph with {num_nodes} nodes and {len(G.edges())} edges\n")
            return data

        except Exception as e:
            print(f"Error loading graph {graph_path}: {str(e)}")
            # Return a minimal valid graph with self-loop
            return Data(
                x=torch.ones(1, 1, dtype=torch.float32),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long)
            )

    def __len__(self):
        return len(self.assembly_ids)

    def __getitem__(self, idx):
        try:
            assembly_id = self.assembly_ids[idx]

            # Check cache first
            if self.cache_images and idx in self.image_cache:
                return self.image_cache[idx]

            # Load assembly image
            assembly_image = Image.open(self.assembly_images[idx]).convert('RGB')
            assembly_image = self.transform(assembly_image)

            # Load part images and convert to tensor immediately
            part_images = []
            for part_path in self.part_images[idx]:
                part_img = Image.open(part_path).convert('RGB')
                part_images.append(self.transform(part_img))
            part_images = torch.stack(part_images) if part_images else torch.zeros(0, 3, 224, 224)

            # Load graph
            graph = self.load_graph(self.graph_paths[idx])

            result = {
                'assembly_image': assembly_image,
                'part_images': part_images,
                'graph': graph,
                'assembly_id': assembly_id
            }

            # Cache the result if enabled
            if self.cache_images:
                self.image_cache[idx] = result

            return result

        except Exception as e:
            print(f"Error loading assembly {idx}: {str(e)}")
            raise

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable number of parts
    """
    # Get the maximum number of parts in this batch
    max_parts = max(len(item['part_images']) for item in batch)

    batch_assembly_ids = []
    batch_assembly_images = []
    batch_part_images = []
    batch_graphs = []
    batch_part_masks = []  # To keep track of which parts are padding

    for item in batch:
        batch_assembly_ids.append(item['assembly_id'])
        batch_assembly_images.append(item['assembly_image'])

        # Get number of parts in this assembly
        num_parts = len(item['part_images'])

        # Create mask (1 for real parts, 0 for padding)
        mask = torch.ones(num_parts)
        if num_parts < max_parts:
            # Pad with zeros
            mask = torch.cat([mask, torch.zeros(max_parts - num_parts)])

            # Pad part images by repeating the last part
            padding = item['part_images'][-1].unsqueeze(0).repeat(max_parts - num_parts, 1, 1, 1)
            padded_parts = torch.cat([item['part_images'], padding], dim=0)
        else:
            padded_parts = item['part_images']

        batch_part_images.append(padded_parts)
        batch_graphs.append(item['graph'])
        batch_part_masks.append(mask)

    return {
        'assembly_id': batch_assembly_ids,
        'assembly_image': torch.stack(batch_assembly_images),
        'part_images': torch.stack(batch_part_images),
        'graph': batch_graphs,  # Handled by PyG's DataLoader
        'part_masks': torch.stack(batch_part_masks)  # Masks for valid parts
    }