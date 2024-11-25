import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

def get_assembly_id_from_path(image_path):
    """Extract assembly ID from image path"""
    path_parts = os.path.normpath(image_path).split(os.sep)
    # Look for a part that's a number (assembly ID)
    assembly_id = next((part for part in path_parts if part.isdigit()), None)
    if assembly_id is None:
        raise ValueError(f"Could not find assembly ID in path: {image_path}")
    return assembly_id

def visualize_results(query_image_path, results, data_dir, num_results=5):
    """Visualize assembly query and top retrieved assemblies"""
    num_images = min(num_results + 1, len(results) + 1)
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))

    # Handle single subplot case
    if num_images == 1:
        axes = [axes]

    # Show query image
    query_img = Image.open(query_image_path)
    axes[0].imshow(query_img)
    query_id = get_assembly_id_from_path(query_image_path)
    axes[0].set_title(f'Query Image\n(Assembly {query_id})')
    axes[0].axis('off')

    # Show retrieved images
    for idx, result in enumerate(results[:num_results], 1):
        assembly_id = result['assembly_id']
        similarity = result['similarity']

        # Construct path to retrieved assembly image
        retrieved_img_path = os.path.join(
            data_dir,
            str(assembly_id),
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
    save_path = os.path.join(save_dir, f'query_{query_id}_results.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\nVisualization saved to {save_path}")
    plt.close()

def visualize_part_query_results(query_image_path, results, data_dir, num_results=10):
    """Visualize part query and matching parts from top assemblies"""
    num_assemblies = min(num_results, len(results))

    # Adjust figure size based on number of results
    fig_width = 4 * (num_assemblies + 1)  # +1 for query column
    fig, axes = plt.subplots(2, num_assemblies + 1, figsize=(fig_width, 8))

    # Show query part
    query_img = Image.open(query_image_path)
    axes[0, 0].imshow(query_img)
    query_id = get_assembly_id_from_path(query_image_path)
    axes[0, 0].set_title('Query Part\n(from Assembly {})'.format(query_id))
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')  # Empty space below query

    # Show results
    for idx, result in enumerate(results[:num_results], 1):
        assembly_id = result['assembly_id']
        similarity = result['similarity']
        part_idx = result['part_idx']

        # Show full assembly
        assembly_img_path = os.path.join(
            data_dir,
            str(assembly_id),
            'images',
            f'{assembly_id}_full_assembly.png'
        )
        if os.path.exists(assembly_img_path):
            img = Image.open(assembly_img_path)
            axes[0, idx].imshow(img)
            is_same = "✓" if str(assembly_id) == str(query_id) else ""
            axes[0, idx].set_title(f'Assembly {assembly_id} {is_same}\nSimilarity: {similarity:.4f}')
        else:
            axes[0, idx].text(0.5, 0.5, 'Assembly image not found', ha='center')
        axes[0, idx].axis('off')

        # Show matching part
        part_img_path = os.path.join(
            data_dir,
            str(assembly_id),
            'images',
            result['part_name'] + '.png'
        )

        if os.path.exists(part_img_path):
            part_img = Image.open(part_img_path)
            axes[1, idx].imshow(part_img)
            axes[1, idx].set_title(f'Matching Part\n{result["part_name"]}')
        else:
            axes[1, idx].text(0.5, 0.5, 'Part image not found', ha='center')
        axes[1, idx].axis('off')

    plt.tight_layout()

    # Save visualization
    save_dir = "retrieval_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'part_query_{query_id}_results.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\nVisualization saved to {save_path}")
    plt.close()