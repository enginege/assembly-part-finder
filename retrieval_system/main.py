import logging
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
    logging.debug("\nSystem Information:")
    logging.debug(f"Python version: {sys.version}")
    logging.debug(f"Platform: {platform.platform()}")
    logging.debug(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logging.info(f"CUDA available: Yes")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("CUDA available: No")

    try:
        import faiss
        if hasattr(faiss, 'StandardGpuResources'):
            logging.info("FAISS-GPU available: Yes")
        else:
            logging.info("FAISS-GPU available: No (CPU version installed)")
    except ImportError:
        logging.error("FAISS not installed")
        sys.exit(1)

def test_retrieval(retrieval_system, dataset, num_queries=5):
    """Test the retrieval system with random queries"""
    logging.debug("\nTesting retrieval system...")

    # Get random indices for testing
    test_indices = np.random.choice(len(dataset), num_queries, replace=False)

    for idx in test_indices:
        # Get test assembly
        test_sample = dataset[idx]
        assembly_id = test_sample['assembly_id']
        query_image = test_sample['assembly_image'].unsqueeze(0)

        logging.debug(f"\nQuery assembly ID: {assembly_id}")
        logging.debug(f"Number of parts: {len(test_sample['part_images'])}")

        # Get retrievals
        results = retrieval_system.retrieve(query_image, k=5)

        logging.debug("\nTop 5 similar assemblies:")
        for rank, result in enumerate(results, 1):
            logging.debug(f"Rank {rank}: Assembly {result['assembly_id']} (similarity: {result['similarity']:.4f})")

        logging.debug("-" * 50)

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
    logging.info(f"\nModel saved to {save_path}")

def load_model(device):
    """Load the trained model"""
    if os.path.exists(os.path.join("./saved_models", "best_model.pth")):
        model_path = os.path.join("./saved_models", "best_model.pth")
        index_path = os.path.join("./saved_models", "best_model_index.pkl")
    else:
        model_path = os.path.join("./saved_models", "retrieval_model.pth")
        index_path = os.path.join("./saved_models", "retrieval_index.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Handle both old and new checkpoint formats
    if 'model_config' in checkpoint:
        embedding_dim = checkpoint['model_config']['embedding_dim']
    else:
        embedding_dim = checkpoint.get('embedding_dim', 256)

    model = RetrievalModel(embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logging.info(f"\nModel loaded from {model_path}")
    return model, index_path

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
        logging.debug("-" * 80)
        if query_type == 'assembly':
            logging.debug(f"{'Rank':<6}{'Assembly ID':<12}{'Similarity':<10}{'Self-Match':<10}")
        else:
            logging.debug(f"{'Rank':<6}{'Assembly ID':<12}{'Similarity':<10}{'Part Name':<30}{'Same Assembly':<12}")
        logging.debug("-" * 80)

        for rank, result in enumerate(results, 1):
            assembly_id = result['assembly_id']
            if query_type == 'assembly':
                is_self = "✓" if assembly_id == query_id else ""
                logging.debug(f"{rank:<6}{assembly_id:<12}{result['similarity']:.4f}     {is_self:^10}")
            else:
                is_same_assembly = "✓" if str(assembly_id) == str(query_id) else ""
                logging.debug(f"{rank:<6}{assembly_id:<12}{result['similarity']:.4f}     "
                      f"{result['part_name']:<30}{is_same_assembly:^12}")

        logging.debug("\nStatistics:")
        similarities = [r['similarity'] for r in results]
        logging.debug(f"Average similarity: {np.mean(similarities):.4f}")
        logging.debug(f"Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")

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
    logging.debug(f"Using device: {device}")  # Debug print

    try:
        if args.mode == 'train':
            logging.debug("Starting training mode...")

            # Create saved_models directory at the start
            os.makedirs("./saved_models", exist_ok=True)

            if not args.data_dir:
                raise ValueError("Training mode requires --data_dir argument")

            logging.debug(f"Loading dataset from {args.data_dir}")  # Debug print
            # Create dataset
            train_dataset = AssemblyDataset(args.data_dir, cache_embeddings=True, is_training=True)
            val_dataset = AssemblyDataset(args.data_dir, cache_embeddings=True, is_training=False)

            logging.debug(f"Dataset loaded with {len(train_dataset)} samples")  # Debug print
            # Split into train and validation sets
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,  # Reduced from 2
                pin_memory=False,  # Already false, good
                persistent_workers=False,  # Changed from True
                prefetch_factor=None,  # Remove prefetching
                collate_fn=custom_collate_fn
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,  # Reduced from 2
                pin_memory=False,  # Already false, good
                persistent_workers=False,  # Changed from True
                prefetch_factor=None,  # Remove prefetching
                collate_fn=custom_collate_fn
            )

            logging.info(f"Dataset loaded with {len(train_dataset)} samples")  # Debug print

            # Initialize model and trainer
            logging.info("Initializing model and trainer...")  # Debug print
            model = RetrievalModel(embedding_dim=256).to(device)
            trainer = ModelTrainer(model, device, part_batch_size=2)

            logging.info(f"Starting training for {args.epochs} epochs...")  # Debug print
            # Train model
            trainer.train(train_loader, val_loader, args.epochs)

            logging.info("\nTraining completed, saving model...")
            #save_model(model)  # Use the new save_model function

            # Build and save index
            logging.info("\nBuilding retrieval index...")
            # retrieval_system = RetrievalSystem(model, device)
            # retrieval_system.build_index(train_loader)

            # index_path = os.path.join("./saved_models", "retrieval_index.pkl")
            # retrieval_system.save_index(index_path)

            # Add after model initialization and before training loop
            logging.info("Precomputing embeddings for training set...")
            train_dataset.dataset.precompute_embeddings(model)  # Access underlying dataset through random split
            logging.info("Precomputing embeddings for validation set...")
            val_dataset.dataset.precompute_embeddings(model)

            # Report cache statistics
            train_dataset.dataset.report_cache_stats()
            val_dataset.dataset.report_cache_stats()

        elif args.mode == 'query':
            if not args.query_image or not args.query_type:
                raise ValueError("Query mode requires --query_image and --query_type arguments")

            # Load model and index
            model, index_path = load_model(device)  # Get both model and index path
            retrieval_system = RetrievalSystem(model, device)
            retrieval_system.load_index(index_path)

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
        logging.error(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    logging.debug("Script started")  # Debug print
    main()
    logging.debug("Script completed")  # Debug print