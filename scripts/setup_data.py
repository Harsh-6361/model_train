"""Setup sample data for testing."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_iris


def setup_iris_dataset():
    """Set up Iris dataset for tabular model testing."""
    print("Setting up Iris dataset...")
    
    # Load Iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['species'] = iris.target
    
    # Map target to species names
    species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species'].map(species_names)
    
    # Save to CSV
    output_path = Path('data/sample/iris.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved Iris dataset to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Classes: {df['species'].unique()}")
    
    return output_path


def setup_synthetic_images():
    """Set up synthetic image dataset for vision model testing."""
    print("Setting up synthetic image dataset...")
    
    output_dir = Path('data/sample/images')
    
    # Create class directories
    classes = ['class_0', 'class_1', 'class_2']
    for class_name in classes:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate 10 synthetic images per class
        for i in range(10):
            # Create a simple colored image
            if class_name == 'class_0':
                color = (255, 0, 0)  # Red
            elif class_name == 'class_1':
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Blue
            
            # Add some noise
            img_array = np.ones((64, 64, 3), dtype=np.uint8) * color
            noise = np.random.randint(-20, 20, (64, 64, 3))
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            # Save image
            img = Image.fromarray(img_array)
            img_path = class_dir / f"{class_name}_{i:03d}.jpg"
            img.save(img_path)
        
        print(f"Created 10 images for {class_name}")
    
    print(f"Saved synthetic images to {output_dir}")
    return output_dir


def setup_sample_data():
    """Set up all sample datasets."""
    print("=== Setting up sample data ===")
    
    try:
        # Setup Iris dataset
        iris_path = setup_iris_dataset()
        print(f"✓ Iris dataset ready: {iris_path}")
        
        # Setup synthetic images
        images_path = setup_synthetic_images()
        print(f"✓ Synthetic images ready: {images_path}")
        
        print("\n=== Sample data setup complete ===")
        print("\nYou can now train models using:")
        print("  Tabular: python scripts/train.py --model-type tabular")
        print("  Vision:  python scripts/train.py --model-type vision")
        
    except Exception as e:
        print(f"Error setting up sample data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    setup_sample_data()
