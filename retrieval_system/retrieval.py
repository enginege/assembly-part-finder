import logging
import faiss
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pickle
from PIL import Image
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from .visualization import visualize_part_query_results, get_assembly_id_from_path
import networkx as nx
from torch_geometric.data import Data

class RetrievalSystem:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.assembly_embeddings = []
        self.part_embeddings = []
        self.graph_embeddings = []
        self.assembly_ids = []
        self.part_names = []

    def process_graphs(self, graphs):
        """Process a list of graphs"""
        graph_embeddings_list = []

        for graph in graphs:
            # Move individual graph to device
            graph_data = graph.to(self.device)

            with torch.no_grad():
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

    def build_index(self, dataloader):
        """Build the retrieval index from the dataset"""
        logging.debug("\nBuilding retrieval index...")
        self.model.eval()
        assembly_embeddings_list = []
        part_embeddings_list = []
        graph_embeddings_list = []
        assembly_ids_list = []
        part_names_list = []

        batch_size = 1  # Smaller batch size for processing

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing assemblies"):
                try:
                    # Process in smaller chunks
                    for i in range(0, len(batch['assembly_image']), batch_size):
                        # Clear cache before processing each chunk
                        torch.cuda.empty_cache()

                        end_idx = min(i + batch_size, len(batch['assembly_image']))

                        # Process assembly images
                        assembly_imgs = batch['assembly_image'][i:end_idx].to(self.device)
                        assembly_embeddings = self.model.encode_assembly(assembly_imgs)
                        assembly_embeddings_list.append(assembly_embeddings.cpu())

                        # Process parts
                        for j in range(i, end_idx):
                            parts = batch['part_images'][j].to(self.device)
                            # Process parts in even smaller chunks if needed
                            part_embeddings = []
                            for k in range(0, len(parts), 4):  # Process 4 parts at a time
                                part_chunk = parts[k:k+4]
                                part_emb = self.model.encode_part(part_chunk)
                                part_embeddings.append(part_emb.cpu())
                            part_embeddings = torch.cat(part_embeddings, dim=0)
                            part_embeddings_list.append(part_embeddings)

                            # Process graphs
                            graph = batch['graph'][j].to(self.device)
                            graph_emb = self.model.encode_graph(graph.x, graph.edge_index)
                            graph_embeddings_list.append(graph_emb.cpu())

                            assembly_ids_list.append(batch['assembly_id'][j])

                        # Clear GPU memory after each chunk
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.error(f"\nOOM error during index building. Skipping batch...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        # Save index immediately after building
        self.assembly_embeddings = torch.cat(assembly_embeddings_list, dim=0)
        self.graph_embeddings = torch.cat(graph_embeddings_list, dim=0)
        self.part_embeddings = part_embeddings_list
        self.assembly_ids = assembly_ids_list

    def retrieve(self, query_image, k=5):
        self.model.eval()

        with torch.no_grad():
            # Get query embedding
            query_img = query_image.unsqueeze(0).to(self.device)
            query_embedding = self.model.encode_assembly(query_img)

            # Compute similarities
            similarities = F.cosine_similarity(
                query_embedding.cpu(),
                self.assembly_embeddings,
                dim=1
            )

            # Get top-k matches
            top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(self.assembly_ids)))

            results = []
            for i, idx in enumerate(top_k_indices):
                results.append({
                    'assembly_id': self.assembly_ids[idx],
                    'similarity': top_k_values[i].item()
                })

            return results

    def save_index(self, path):
        """Save the index to a file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'assembly_embeddings': self.assembly_embeddings,
                'part_embeddings': self.part_embeddings,
                'graph_embeddings': self.graph_embeddings,
                'assembly_ids': self.assembly_ids,
                'part_names': self.part_names
            }, f)
        logging.debug(f"Index saved to {path}")

    def load_index(self, path):
        """Load the index from a file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # Move all embeddings to device immediately after loading
            self.assembly_embeddings = torch.tensor(data['assembly_embeddings']).float().to(self.device)
            self.graph_embeddings = torch.tensor(data['graph_embeddings']).float().to(self.device)
            # Part embeddings is a list of tensors, handle separately
            self.part_embeddings = [torch.tensor(emb).float().to(self.device) for emb in data['part_embeddings']]
            self.assembly_ids = data['assembly_ids']
            self.part_names = data['part_names']
        logging.debug(f"Index loaded from {path}")

    def retrieve_by_assembly(self, query_image, k=10):
        """Retrieve similar assemblies using an assembly image"""
        self.model.eval()
        with torch.no_grad():
            # Add batch dimension to query image
            query_img = query_image.unsqueeze(0).to(self.device)
            query_embedding = self.model.encode_assembly(query_img)

            # Ensure assembly_embeddings is a tensor
            if not torch.is_tensor(self.assembly_embeddings):
                self.assembly_embeddings = torch.tensor(self.assembly_embeddings).float()

            similarities = F.cosine_similarity(
                query_embedding.cpu().float(),
                self.assembly_embeddings,
                dim=1
            )

            return self._get_top_k_results(similarities, k)

    def get_part_name(self, assembly_id, part_idx, data_dir):
        """Get the actual filename from the directory without assumptions about naming"""
        assembly_dir = os.path.join(data_dir, str(assembly_id), 'images')

        try:
            # List all files in the directory
            files = os.listdir(assembly_dir)
            # Filter out full_assembly.png and SOLID.png
            part_files = [f for f in files if not (f.endswith('full_assembly.png') or f == 'SOLID.png')]
            # Sort files to ensure consistent ordering
            part_files.sort()

            # If we have parts and the index is valid
            if part_files and part_idx < len(part_files):
                # Return the filename without extension
                return os.path.splitext(part_files[part_idx])[0]

        except Exception as e:
            logging.error(f"Error reading directory {assembly_dir}: {str(e)}")

        return f'Part {part_idx}'  # Fallback name if something goes wrong

    def retrieve_by_part(self, query_image, query_image_path, k=10, exclude_query_assembly=False,
                    max_parts_per_assembly=2, similarity_threshold=0.7, data_dir=None):
        """Retrieve similar parts using multi-modal features"""
        if not data_dir:
            raise ValueError("data_dir must be provided for part retrieval")

        query_id = get_assembly_id_from_path(query_image_path)
        self.model.eval()

        with torch.no_grad():
            # Get query embeddings
            query_img = query_image.unsqueeze(0).to(self.device)
            query_part_embedding = F.normalize(self.model.encode_part(query_img).float(), dim=1)

            # Get query assembly image and graph
            query_assembly_dir = os.path.join(data_dir, str(query_id), 'images')
            query_assembly_path = os.path.join(query_assembly_dir, f'{query_id}_full_assembly.png')
            query_graph_path = os.path.join(data_dir, str(query_id), f'{query_id}_assembly.graphml')

            # Load and encode query assembly and graph
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            query_assembly_img = Image.open(query_assembly_path).convert('RGB')
            query_assembly_img = transform(query_assembly_img).unsqueeze(0).to(self.device)
            query_assembly_embedding = F.normalize(self.model.encode_assembly(query_assembly_img).float(), dim=1)

            query_graph = self.load_graph(query_graph_path).to(self.device)
            query_graph_embedding = F.normalize(self.model.encode_graph(query_graph.x, query_graph.edge_index).float(), dim=1)

            all_part_similarities = []

            for assembly_id, part_embeddings in zip(self.assembly_ids, self.part_embeddings):
                if exclude_query_assembly and str(assembly_id) == str(query_id):
                    continue

                # Get assembly and graph embeddings for current assembly
                assembly_idx = self.assembly_ids.index(assembly_id)
                assembly_embedding = self.assembly_embeddings[assembly_idx].unsqueeze(0)  # Already on device
                graph_embedding = self.graph_embeddings[assembly_idx].unsqueeze(0)  # Already on device

                # Part embeddings are already on device, just normalize
                part_embeddings = F.normalize(part_embeddings, dim=1)

                # Compute similarities (all tensors now on same device)
                part_similarities = F.cosine_similarity(
                    query_part_embedding,
                    part_embeddings,
                    dim=1
                )

                assembly_similarity = F.cosine_similarity(
                    query_assembly_embedding,
                    assembly_embedding,
                    dim=1
                )

                graph_similarity = F.cosine_similarity(
                    query_graph_embedding,
                    graph_embedding,
                    dim=1
                )

                # Get parts directory
                assembly_dir = os.path.join(data_dir, str(assembly_id), 'images')
                if not os.path.exists(assembly_dir):
                    continue

                part_files = [f for f in os.listdir(assembly_dir)
                             if not (f.endswith('full_assembly.png') or f == 'SOLID.png')]
                part_files.sort()

                # Combine similarities with weights
                for part_idx, part_similarity in enumerate(part_similarities):
                    if part_idx >= len(part_files):
                        break

                    # Weighted combination of similarities
                    combined_similarity = (
                        0.6 * part_similarity.item() +  # Higher weight for part similarity
                        0.2 * assembly_similarity.item() +  # Lower weight for assembly context
                        0.2 * graph_similarity.item()  # Lower weight for structural similarity
                    )

                    if combined_similarity > similarity_threshold:
                        part_name = os.path.splitext(part_files[part_idx])[0]
                        all_part_similarities.append({
                            'assembly_id': str(assembly_id),
                            'similarity': combined_similarity,
                            'part_similarity': part_similarity.item(),
                            'assembly_similarity': assembly_similarity.item(),
                            'graph_similarity': graph_similarity.item(),
                            'part_idx': part_idx,
                            'part_name': part_name
                        })

            # Sort by combined similarity
            all_part_similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # Limit results per assembly
            limited_results = []
            assembly_count = {}

            for result in all_part_similarities:
                assembly_id = result['assembly_id']
                if assembly_id not in assembly_count:
                    assembly_count[assembly_id] = 0

                if assembly_count[assembly_id] < max_parts_per_assembly:
                    limited_results.append(result)
                    assembly_count[assembly_id] += 1

                if len(limited_results) >= k:
                    break

            return limited_results[:k]

    def load_graph(self, graph_path):
        """Load graph from file"""
        try:
            G = nx.read_graphml(graph_path)
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, node_mapping)

            edge_index = []
            for edge in G.edges():
                edge_index.append([edge[0], edge[1]])
                edge_index.append([edge[1], edge[0]])

            edge_index = torch.tensor(edge_index).t().contiguous()
            x = torch.ones(G.number_of_nodes(), 1, dtype=torch.float32)

            return Data(x=x, edge_index=edge_index.long())
        except Exception as e:
            logging.error(f"Error loading graph {graph_path}: {str(e)}")
            return Data(
                x=torch.ones(1, 1, dtype=torch.float32),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long)
            )

    def _get_top_k_results(self, similarities, k):
        """Helper function to get top-k results"""
        top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(self.assembly_ids)))

        results = []
        for i, idx in enumerate(top_k_indices):
            results.append({
                'assembly_id': self.assembly_ids[idx],
                'similarity': top_k_values[i].item()
            })

        return results

    def compute_embeddings(self, image_tensor):
        """Compute embeddings for an image tensor"""
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            return self.model.encode_assembly(image_tensor.to(self.device)).cpu()

def visualize_results(query_image_path, results, data_dir, num_results=5):
    """Visualize query image and top retrieved results"""
    num_images = min(num_results + 1, len(results) + 1)
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))

    # Show query image
    query_img = Image.open(query_image_path)
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image\n(Assembly {})'.format(
        query_image_path.split(os.sep)[-3]
    ))
    axes[0].axis('off')

    # Show retrieved images
    for idx, result in enumerate(results[:num_results], 1):
        assembly_id = result['assembly_id']
        similarity = result['similarity']

        # Construct path to retrieved assembly image
        retrieved_img_path = os.path.join(
            data_dir,
            assembly_id,
            'images',
            f'{assembly_id}_full_assembly.png'
        )

        if os.path.exists(retrieved_img_path):
            img = Image.open(retrieved_img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f'Rank {idx}\nAssembly {assembly_id}\nSimilarity: {similarity:.4f}')
        else:
            axes[idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[idx].set_title(f'Rank {idx}\nAssembly {assembly_id}\n(missing image)')
        axes[idx].axis('off')

    plt.tight_layout()

    # Save the visualization
    save_dir = "retrieval_results"
    os.makedirs(save_dir, exist_ok=True)
    query_id = query_image_path.split(os.sep)[-3]
    save_path = os.path.join(save_dir, f'query_{query_id}_results.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    logging.debug(f"\nVisualization saved to {save_path}")
    plt.close()

def query_system(retrieval_system, image_path, query_type='assembly', k=10,
                data_dir=None, exclude_query_assembly=False, max_parts_per_assembly=5):
    """Query the system with either an assembly or part image"""
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        query_image = transform(image).float()

        # Get assembly ID from image path
        query_id = image_path.split(os.sep)[-3]
        logging.debug(f"\nQuerying with {query_type} image: {image_path}")
        logging.debug(f"Query Assembly ID: {query_id}")

        if query_type == 'assembly':
            results = retrieval_system.retrieve_by_assembly(query_image, k=k)
        else:  # part
            results = retrieval_system.retrieve_by_part(
                query_image,
                image_path,
                k=k,
                exclude_query_assembly=exclude_query_assembly,
                max_parts_per_assembly=max_parts_per_assembly,
                data_dir=data_dir
            )

        logging.debug("\nRetrieval Results:")
        logging.debug("-" * 100)
        if query_type == 'part':
            logging.debug(f"{'Rank':<6}{'Assembly ID':<12}{'Combined':<10}{'Part':<10}{'Assembly':<10}{'Graph':<10}{'Part Name':<30}")
            logging.debug("-" * 100)

            for rank, result in enumerate(results, 1):
                logging.debug(f"{rank:<6}{result['assembly_id']:<12}"
                      f"{result['similarity']:<10.4f}"
                      f"{result['part_similarity']:<10.4f}"
                      f"{result['assembly_similarity']:<10.4f}"
                      f"{result['graph_similarity']:<10.4f}"
                      f"{result['part_name']:<30}")
        else:
            logging.debug(f"{'Rank':<6}{'Assembly ID':<12}{'Similarity':<12}{'Part Name':<30}{'Same Assembly':<10}")
            logging.debug("-" * 80)

            for rank, result in enumerate(results, 1):
                assembly_id = result['assembly_id']
                similarity = result['similarity']
                part_name = os.path.basename(image_path) if 'part_name' not in result else result['part_name']
                is_same_assembly = "✓" if str(assembly_id) == str(query_id) else " "

                logging.debug(f"{rank:<6}{assembly_id:<12}{similarity:<12.4f}{part_name:<30}{is_same_assembly:^10}")

        logging.debug("\nStatistics:")
        similarities = [r['similarity'] for r in results]
        logging.debug(f"Average similarity: {np.mean(similarities):.4f}")
        logging.debug(f"Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")

        if query_id in [str(r['assembly_id']) for r in results]:
            rank = [str(r['assembly_id']) for r in results].index(query_id) + 1
            logging.debug(f"Query assembly found at rank {rank}")
        else:
            logging.warning("Query assembly not found in top results")

        # Visualize results if data_dir is provided
        if data_dir:
            if query_type == 'assembly':
                visualize_results(image_path, results, data_dir)
            else:
                visualize_part_query_results(image_path, results, data_dir)

        return results

    except Exception as e:
        logging.error(f"Error processing query image: {str(e)}")
        logging.error(f"Error Image path: {image_path}")
        raise