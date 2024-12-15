from PIL import Image
import os
from tqdm import tqdm
import concurrent.futures
from functools import partial

def preprocess_image(image_path, output_path, target_size=(480, 480)):
    """
    Preprocess an image by:
    1. Center cropping
    2. Converting to grayscale
    3. Resizing to target size
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save processed image
        target_size (tuple): Desired output size (width, height)
    
    Returns:
        tuple: Original size and new size in KB, or None if processing failed
    """
    try:
        # Open the image
        img = Image.open(image_path)

        # Get original dimensions
        width, height = img.size

        # Calculate dimensions for center crop
        min_dim = min(width, height)
        # Reduce the crop size by a small percentage (e.g., 5%)
        crop_reduction = 0.05  # 5%
        adjusted_min_dim = int(min_dim * (1 - crop_reduction))

        left = (width - adjusted_min_dim) // 2
        top = (height - adjusted_min_dim) // 2
        right = left + adjusted_min_dim
        bottom = top + adjusted_min_dim

        # Apply center crop
        img_cropped = img.crop((left, top, right, bottom))

        # Convert to grayscale
        img_gray = img_cropped.convert('L')

        # Resize to target size
        img_resized = img_gray.resize(target_size, Image.Resampling.LANCZOS)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the processed image
        img_resized.save(output_path, optimize=True, quality=85)

        # Get file sizes
        original_size = os.path.getsize(image_path) / 1024  # KB
        new_size = os.path.getsize(output_path) / 1024  # KB
        
        return original_size, new_size

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_single_image(image_path, target_size):
    """
    Process a single image and return the results
    """
    file_ext = os.path.splitext(image_path)[1]
    temp_output_path = f"{os.path.splitext(image_path)[0]}_temp{file_ext}"
    
    result = preprocess_image(image_path, temp_output_path, target_size)
    
    if result:
        original_size, new_size = result
        # Replace original file with processed version
        os.replace(temp_output_path, image_path)
        return True, original_size, new_size
    else:
        # Clean up temp file if it exists
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return False, 0, 0

def process_directory_recursive(input_dir, target_size=(480, 480), max_workers=None):
    """
    Recursively process all images in a directory and its subdirectories,
    replacing original files with processed versions using multiple threads
    
    Args:
        input_dir (str): Input directory containing images
        target_size (tuple): Desired output size (width, height)
        max_workers (int): Maximum number of worker threads (None = CPU count)
    """
    # Supported image formats
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Statistics
    total_original_size = 0
    total_new_size = 0
    processed_files = 0
    failed_files = 0

    # Get all image files recursively
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                image_files.append(os.path.join(root, file))

    # Create a partial function with the target size
    process_func = partial(process_single_image, target_size=target_size)

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and wrap with tqdm for progress
        futures = list(tqdm(executor.map(process_func, image_files), 
                          total=len(image_files),
                          desc="Processing images"))
        
        # Process results
        for success, original_size, new_size in futures:
            if success:
                total_original_size += original_size
                total_new_size += new_size
                processed_files += 1
            else:
                failed_files += 1

    # Print summary statistics
    print("\nProcessing Summary:")
    print(f"Successfully processed files: {processed_files}")
    print(f"Failed files: {failed_files}")
    print(f"Total original size: {total_original_size:.2f} KB ({total_original_size/1024:.2f} MB)")
    print(f"Total new size: {total_new_size:.2f} KB ({total_new_size/1024:.2f} MB)")
    if total_original_size > 0:  # Prevent division by zero
        print(f"Total size reduction: {((total_original_size - total_new_size) / total_original_size * 100):.2f}%")
        print(f"Total space saved: {(total_original_size - total_new_size):.2f} KB ({(total_original_size - total_new_size)/1024:.2f} MB)")
    else:
        print("No files were successfully processed")

# Example usage
if __name__ == "__main__":
    input_directory = "dataset"
    process_directory_recursive(input_directory, max_workers=16)
if __name__ == "__main__":
    input_directory = "dataset"
    process_directory_recursive(input_directory, max_workers=16)