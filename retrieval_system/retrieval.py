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
from .visualization import visualize_part_query_results

class RetrievalSystem:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.assembly_embeddings = []
        self.part_embeddings = []
        self.graph_embeddings = []
        self.assembly_ids = []

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
        print("\nBuilding retrieval index...")
        self.model.eval()
        assembly_embeddings_list = []
        part_embeddings_list = []
        graph_embeddings_list = []
        assembly_ids_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing assemblies"):
                # Process assembly images
                assembly_imgs = batch['assembly_image'].to(self.device)
                assembly_embeddings = self.model.encode_assembly(assembly_imgs)
                assembly_embeddings_list.append(assembly_embeddings.cpu())

                # Process graphs
                graph_embeddings = self.process_graphs(batch['graph'])
                graph_embeddings_list.append(graph_embeddings.cpu())

                # Process parts
                for parts in batch['part_images']:
                    parts = parts.to(self.device)
                    part_embeddings = self.model.encode_part(parts)
                    part_embeddings_list.append(part_embeddings.cpu())

                assembly_ids_list.extend(batch['assembly_id'])

        # Concatenate all embeddings
        self.assembly_embeddings = torch.cat(assembly_embeddings_list, dim=0)
        self.graph_embeddings = torch.cat(graph_embeddings_list, dim=0)
        self.part_embeddings = part_embeddings_list
        self.assembly_ids = assembly_ids_list

        print(f"Index built with {len(self.assembly_ids)} assemblies")

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
                'assembly_ids': self.assembly_ids
            }, f)
        print(f"Index saved to {path}")

    def load_index(self, path):
        """Load the index from a file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.assembly_embeddings = data['assembly_embeddings']
            self.part_embeddings = data['part_embeddings']
            self.graph_embeddings = data['graph_embeddings']
            self.assembly_ids = data['assembly_ids']
        print(f"Index loaded from {path}")

    def retrieve_by_assembly(self, query_image, k=5):
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
            print(f"Error reading directory {assembly_dir}: {str(e)}")

        return f'Part {part_idx}'  # Fallback name if something goes wrong

    def retrieve_by_part(self, query_image, k=5, similarity_threshold=0.8, data_dir=None):
        """Retrieve similar parts from all assemblies, including multiple parts from same assembly"""
        self.model.eval()
        with torch.no_grad():
            # Add batch dimension to query image
            query_img = query_image.unsqueeze(0).to(self.device)
            query_embedding = self.model.encode_part(query_img)

            # Store all part similarities
            all_part_similarities = []

            for assembly_id, part_embeddings in zip(self.assembly_ids, self.part_embeddings):
                if not torch.is_tensor(part_embeddings):
                    part_embeddings = torch.tensor(part_embeddings).to(self.device)

                # Compare with all parts in this assembly
                similarities = F.cosine_similarity(
                    query_embedding,
                    part_embeddings.to(self.device),
                    dim=1
                )

                # Get all parts above threshold
                for part_idx, similarity in enumerate(similarities):
                    similarity = similarity.item()
                    if similarity > similarity_threshold:
                        part_name = self.get_part_name(assembly_id, part_idx, data_dir) if data_dir else f'Part {part_idx}'
                        all_part_similarities.append({
                            'assembly_id': str(assembly_id),
                            'similarity': similarity,
                            'part_idx': part_idx,
                            'part_name': part_name
                        })

            # Sort all parts by similarity
            sorted_results = sorted(all_part_similarities,
                                 key=lambda x: x['similarity'],
                                 reverse=True)

            return sorted_results[:k]

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
    print(f"\nVisualization saved to {save_path}")
    plt.close()

def query_system(retrieval_system, image_path, query_type='assembly', k=10, data_dir=None):
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
        print(f"\nQuerying with {query_type} image: {image_path}")
        print(f"Query Assembly ID: {query_id}")

        if query_type == 'assembly':
            results = retrieval_system.retrieve_by_assembly(query_image, k=k)
        else:  # part
            results = retrieval_system.retrieve_by_part(query_image, k=k)

        print("\nRetrieval Results:")
        print("-" * 80)
        print(f"{'Rank':<6}{'Assembly ID':<12}{'Similarity':<12}{'Part Name':<30}{'Same Assembly':<10}")
        print("-" * 80)

        for rank, result in enumerate(results, 1):
            assembly_id = result['assembly_id']
            similarity = result['similarity']
            part_name = os.path.basename(image_path) if 'part_name' not in result else result['part_name']
            is_same_assembly = "✓" if str(assembly_id) == str(query_id) else " "

            print(f"{rank:<6}{assembly_id:<12}{similarity:<12.4f}{part_name:<30}{is_same_assembly:^10}")

        print("\nStatistics:")
        similarities = [r['similarity'] for r in results]
        print(f"Average similarity: {np.mean(similarities):.4f}")
        print(f"Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")

        if query_id in [str(r['assembly_id']) for r in results]:
            rank = [str(r['assembly_id']) for r in results].index(query_id) + 1
            print(f"Query assembly found at rank {rank}")
        else:
            print("Query assembly not found in top results")

        # Visualize results if data_dir is provided
        if data_dir:
            if query_type == 'assembly':
                visualize_results(image_path, results, data_dir)
            else:
                visualize_part_query_results(image_path, results, data_dir)

        return results

    except Exception as e:
        print(f"Error processing query image: {str(e)}")
        print(f"Image path: {image_path}")
        raise