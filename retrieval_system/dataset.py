import os
import torch
import networkx as nx
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch_geometric.data import Data
from collections import OrderedDict
import psutil
import gc
from tqdm import tqdm

class MemoryAwareCache:
    def __init__(self, max_memory_percent=30, cache_size=100):  # Reduced from 500
        self.max_memory_percent = max_memory_percent
        self.cache_size = cache_size
        self.train_cache = OrderedDict()
        self.val_cache = OrderedDict()
        self.cache_stats = {'hits': 0, 'misses': 0}

    def _check_memory(self):
        """Check if memory usage is above threshold"""
        system_memory = psutil.virtual_memory().percent
        process_memory = psutil.Process().memory_percent()

        if system_memory > self.max_memory_percent or process_memory > self.max_memory_percent:
            print(f"\nSystem memory usage: {system_memory:.1f}%")
            print(f"Process memory usage: {process_memory:.1f}%")
        return system_memory > self.max_memory_percent or process_memory > self.max_memory_percent

    def _trim_cache(self, cache):
        """Remove oldest items from cache until memory is below threshold"""
        while self._check_memory() and cache:
            cache.popitem(last=False)  # Remove oldest item
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()

    def get(self, idx, is_training=True):
        cache = self.train_cache if is_training else self.val_cache
        if idx in cache:
            self.cache_stats['hits'] += 1
            value = cache.pop(idx)
            cache[idx] = value
            return value
        self.cache_stats['misses'] += 1
        return None

    def put(self, idx, value, is_training=True):
        cache = self.train_cache if is_training else self.val_cache

        # Check if we need to trim the cache
        if len(cache) >= self.cache_size or self._check_memory():
            self._trim_cache(cache)

        if idx in cache:
            cache.pop(idx)
        cache[idx] = value

    def clear_unused_cache(self):
        """Aggressively clear cache when memory usage is high"""
        if self._check_memory():
            print("\nMemory usage high, clearing 70% of cache...")  # Increased from 30%
            items_to_remove = int(len(self.train_cache) * 0.7)
            for _ in range(items_to_remove):
                self.train_cache.popitem(last=False)
            if hasattr(self, 'val_cache'):
                val_items_to_remove = int(len(self.val_cache) * 0.7)
                for _ in range(val_items_to_remove):
                    self.val_cache.popitem(last=False)
            torch.cuda.empty_cache()
            gc.collect()

class AssemblyDataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_embeddings=False, is_training=True):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cache_embeddings = cache_embeddings
        self.is_training = is_training
        self.cache_manager = MemoryAwareCache() if cache_embeddings else None

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
            if self.cache_embeddings:
                cached_item = self.cache_manager.get(idx, self.is_training)
                if cached_item is not None:
                    return cached_item

            # Load and process images
            assembly_image = Image.open(self.assembly_images[idx]).convert('RGB')
            assembly_image = self.transform(assembly_image)

            # Load part images
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
                'assembly_id': self.assembly_ids[idx]
            }

            if self.cache_embeddings:
                self.cache_manager.put(idx, result, self.is_training)

            return result
        except Exception as e:
            print(f"Error loading assembly {idx}: {str(e)}")
            raise

    def clear_cache(self):
        """Clear the embedding cache to free memory"""
        if self.cache_embeddings and self.cache_manager:
            self.cache_manager.clear()

    def precompute_embeddings(self, model):
        """Precompute embeddings for all images and graphs"""
        print("\nPrecomputing embeddings...")
        model.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(self))):
                if not self.cache_manager.get(idx, self.is_training):
                    # Force cache clearing every 10 samples
                    if idx % 10 == 0:
                        self.cache_manager.clear_unused_cache()
                        torch.cuda.empty_cache()
                        gc.collect()

                    sample = self.__getitem__(idx)

                    # Move data to model's device
                    assembly_img = sample['assembly_image'].unsqueeze(0).to(model.device)
                    part_imgs = sample['part_images'].to(model.device)
                    graph = sample['graph'].to(model.device)

                    # Compute embeddings
                    assembly_emb = model.encode_assembly(assembly_img)
                    part_embs = model.encode_part(part_imgs)
                    graph_emb = model.encode_graph(graph.x, graph.edge_index)

                    # Clear GPU memory after computing embeddings
                    del assembly_img, part_imgs, graph
                    torch.cuda.empty_cache()

                    # Store embeddings in cache
                    cached_item = {
                        'assembly_embedding': assembly_emb.cpu(),
                        'part_embeddings': part_embs.cpu(),
                        'graph_embedding': graph_emb.cpu(),
                        'assembly_id': sample['assembly_id'],
                        'original_sample': sample
                    }
                    self.cache_manager.put(idx, cached_item, self.is_training)

                    # Clear more aggressively
                    del assembly_emb, part_embs, graph_emb
                    torch.cuda.empty_cache()

                self.cache_manager.clear_unused_cache()

    def report_cache_stats(self):
        """Report cache hit/miss statistics"""
        if self.cache_manager:
            total = self.cache_manager.cache_stats['hits'] + self.cache_manager.cache_stats['misses']
            if total > 0:
                hit_rate = self.cache_manager.cache_stats['hits'] / total * 100
                print(f"\nCache Statistics:")
                print(f"Hits: {self.cache_manager.cache_stats['hits']}")
                print(f"Misses: {self.cache_manager.cache_stats['misses']}")
                print(f"Hit Rate: {hit_rate:.2f}%")

def custom_collate_fn(batch):
    """Memory efficient collate function"""
    batch_assembly_ids = []
    batch_assembly_images = []
    batch_part_images = []
    batch_graphs = []
    batch_part_masks = []

    max_parts = max(len(item['part_images']) for item in batch)

    for item in batch:
        batch_assembly_ids.append(item['assembly_id'])
        batch_assembly_images.append(item['assembly_image'])

        num_parts = len(item['part_images'])
        mask = torch.ones(num_parts)

        if num_parts < max_parts:
            # Use zero padding instead of repeating last part
            padding_shape = (max_parts - num_parts,) + item['part_images'].shape[1:]
            padding = torch.zeros(padding_shape, dtype=item['part_images'].dtype)
            mask = torch.cat([mask, torch.zeros(max_parts - num_parts)])
            padded_parts = torch.cat([item['part_images'], padding], dim=0)
        else:
            padded_parts = item['part_images']

        batch_part_images.append(padded_parts)
        batch_graphs.append(item['graph'])
        batch_part_masks.append(mask)

    # Clean up intermediate tensors
    result = {
        'assembly_id': batch_assembly_ids,
        'assembly_image': torch.stack(batch_assembly_images),
        'part_images': torch.stack(batch_part_images),
        'graph': batch_graphs,
        'part_masks': torch.stack(batch_part_masks)
    }

    # Force garbage collection
    del batch_assembly_images, batch_part_images, batch_part_masks
    torch.cuda.empty_cache()

    return result