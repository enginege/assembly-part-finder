import torch
import argparse
import platform
import sys
import numpy as np
from torch.utils.data import DataLoader
from .dataset import AssemblyDataset, custom_collate_fn
from .retrieval_model import RetrievalModel
from .trainer import ModelTrainer
from .retrieval import RetrievalSystem
import os
from PIL import Image
from torchvision import transforms
from .visualization import (
    visualize_results,
    visualize_part_query_results,
    get_assembly_id_from_path
)
import pickle

def check_system_compatibility():
    """Check and print system information and compatibility"""
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available: No")

    try:
        import faiss
        if hasattr(faiss, 'StandardGpuResources'):
            print("FAISS-GPU available: Yes")
        else:
            print("FAISS-GPU available: No (CPU version installed)")
    except ImportError:
        print("FAISS not installed")
        sys.exit(1)

def test_retrieval(retrieval_system, dataset, num_queries=5):
    """Test the retrieval system with random queries"""
    print("\nTesting retrieval system...")

    # Get random indices for testing
    test_indices = np.random.choice(len(dataset), num_queries, replace=False)

    for idx in test_indices:
        # Get test assembly
        test_sample = dataset[idx]
        assembly_id = test_sample['assembly_id']
        query_image = test_sample['assembly_image'].unsqueeze(0)

        print(f"\nQuery assembly ID: {assembly_id}")
        print(f"Number of parts: {len(test_sample['part_images'])}")

        # Get retrievals
        results = retrieval_system.retrieve(query_image, k=5)

        print("\nTop 5 similar assemblies:")
        for rank, result in enumerate(results, 1):
            print(f"Rank {rank}: Assembly {result['assembly_id']} (similarity: {result['similarity']:.4f})")

        print("-" * 50)

def save_model(model, save_dir="./saved_models"):
    """Save the trained model with its configuration"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "retrieval_model.pth")

    # Create checkpoint dictionary with model configuration
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'embedding_dim': model.embedding_dim,
        'model_config': {
            'embedding_dim': model.embedding_dim
        }
    }

    torch.save(checkpoint, save_path)
    print(f"\nModel saved to {save_path}")

def load_model(device):
    """Load the trained model"""
    model_path = os.path.join("./saved_models", "retrieval_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    embedding_dim = checkpoint['model_config']['embedding_dim']

    model = RetrievalModel(embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nModel loaded from {model_path}")
    return model

def query_system(retrieval_system, image_path, query_type='assembly', k=10,
                data_dir=None, exclude_query_assembly=False, max_parts_per_assembly=2):
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        query_image = transform(image).float()

        # Get assembly ID from image path
        query_id = get_assembly_id_from_path(image_path)
        print(f"\nQuerying with {query_type} image: {image_path}")
        print(f"Query Assembly ID: {query_id}")

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

        print("\nRetrieval Results:")
        print("-" * 80)
        if query_type == 'assembly':
            print(f"{'Rank':<6}{'Assembly ID':<12}{'Similarity':<10}{'Self-Match':<10}")
        else:
            print(f"{'Rank':<6}{'Assembly ID':<12}{'Similarity':<10}{'Part Name':<30}{'Same Assembly':<12}")
        print("-" * 80)

        for rank, result in enumerate(results, 1):
            assembly_id = result['assembly_id']
            if query_type == 'assembly':
                is_self = "✓" if assembly_id == query_id else ""
                print(f"{rank:<6}{assembly_id:<12}{result['similarity']:.4f}     {is_self:^10}")
            else:
                is_same_assembly = "✓" if str(assembly_id) == str(query_id) else ""
                print(f"{rank:<6}{assembly_id:<12}{result['similarity']:.4f}     "
                      f"{result['part_name']:<30}{is_same_assembly:^12}")

        print("\nStatistics:")
        similarities = [r['similarity'] for r in results]
        print(f"Average similarity: {np.mean(similarities):.4f}")
        print(f"Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")

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

def main():
    parser = argparse.ArgumentParser(description="Assembly Part Finder")
    parser.add_argument('--mode', type=str, choices=['train', 'query'], required=True, help='Mode to run the system')
    parser.add_argument('--query_image', type=str, help='Path to the query image (for query mode)')
    parser.add_argument('--query_type', type=str, choices=['assembly', 'part'], default='assembly', help='Type of query')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--max_parts_per_assembly', type=int, default=2, help='Maximum number of parts to retrieve from each assembly')
    parser.add_argument('--exclude_query_assembly', default=False, action='store_true',
                       help='Exclude parts from the same assembly as the query')
    parser.add_argument('--data_dir', type=str, help='Path to the dataset directory (required for training)')

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'train' and not args.data_dir:
        parser.error("Training mode requires --data_dir argument")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  # Debug print

    try:
        if args.mode == 'train':
            print("Starting training mode...")

            # Create saved_models directory at the start
            os.makedirs("./saved_models", exist_ok=True)

            if not args.data_dir:
                raise ValueError("Training mode requires --data_dir argument")

            print(f"Loading dataset from {args.data_dir}")  # Debug print
            # Create dataset
            dataset = AssemblyDataset(args.data_dir, cache_images=True)
            print(f"Dataset loaded with {len(dataset)} samples")  # Debug print
            # Split into train and validation sets
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn
            )

            print(f"Dataset loaded with {len(dataset)} samples")  # Debug print

            # Initialize model and trainer
            print("Initializing model and trainer...")  # Debug print
            model = RetrievalModel(embedding_dim=256).to(device)
            trainer = ModelTrainer(model, device)

            print(f"Starting training for {args.epochs} epochs...")  # Debug print
            # Train model
            trainer.train(train_loader, val_loader, args.epochs)

            print("\nTraining completed, saving model...")
            save_model(model)  # Use the new save_model function

            # Build and save index
            print("\nBuilding retrieval index...")
            retrieval_system = RetrievalSystem(model, device)
            retrieval_system.build_index(train_loader)

            index_path = os.path.join("./saved_models", "retrieval_index.pkl")
            retrieval_system.save_index(index_path)

        elif args.mode == 'query':
            if not args.query_image or not args.query_type:
                raise ValueError("Query mode requires --query_image and --query_type arguments")

            # Load model and index
            model = load_model(device)  # Use the new load_model function
            retrieval_system = RetrievalSystem(model, device)
            retrieval_system.load_index("./saved_models/retrieval_index.pkl")

            # Get data directory from query image path
            data_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.query_image)))

            # Process query
            results = query_system(
                retrieval_system,
                args.query_image,
                args.query_type,
                data_dir=data_dir,
                exclude_query_assembly=args.exclude_query_assembly,
                max_parts_per_assembly=args.max_parts_per_assembly
            )

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    print("Script started")  # Debug print
    main()
    print("Script completed")  # Debug print