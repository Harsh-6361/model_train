"""
YOLO Data Adapter
- Support YOLO format (txt annotations)
- Support COCO format (JSON)
- Support Pascal VOC format (XML)
- Auto-conversion between formats
- Dataset splitting and validation
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import yaml
from collections import defaultdict


class YOLODataAdapter:
    """
    Handles YOLO dataset formats and conversions
    
    Supported formats:
    - YOLO: .txt files with class x_center y_center width height
    - COCO: JSON with bounding boxes
    - Pascal VOC: XML annotation files
    """
    
    def __init__(self, root_dir: str):
        """
        Initialize data adapter
        
        Args:
            root_dir: Root directory for dataset
        """
        self.root_dir = Path(root_dir)
        self.annotations = []
        self.images = []
        self.categories = []
        self.format = None
    
    def load(self, path: Union[str, Path], format: str = 'yolo'):
        """
        Load dataset from specified format
        
        Args:
            path: Path to dataset
            format: Dataset format ('yolo', 'coco', 'voc')
        
        Returns:
            self for method chaining
        """
        path = Path(path)
        self.format = format.lower()
        
        if self.format == 'yolo':
            self._load_yolo(path)
        elif self.format == 'coco':
            self._load_coco(path)
        elif self.format == 'voc':
            self._load_voc(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return self
    
    def _load_yolo(self, path: Path):
        """Load YOLO format dataset"""
        # YOLO format: one txt file per image
        # Each line: class x_center y_center width height (normalized)
        label_dir = path / 'labels' if (path / 'labels').exists() else path
        image_dir = path / 'images' if (path / 'images').exists() else path
        
        for label_file in label_dir.glob('*.txt'):
            image_name = label_file.stem
            
            # Find corresponding image
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = image_dir / f"{image_name}{ext}"
                if image_path.exists():
                    self.images.append(str(image_path))
                    break
            
            # Parse annotations
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        self.annotations.append({
                            'image': image_name,
                            'class': int(parts[0]),
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4]),
                        })
    
    def _load_coco(self, path: Path):
        """Load COCO format dataset"""
        annotation_file = path if path.is_file() else path / 'annotations.json'
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.images = data.get('images', [])
        self.annotations = data.get('annotations', [])
        self.categories = data.get('categories', [])
    
    def _load_voc(self, path: Path):
        """Load Pascal VOC format dataset"""
        annotation_dir = path / 'Annotations' if (path / 'Annotations').exists() else path
        image_dir = path / 'JPEGImages' if (path / 'JPEGImages').exists() else path
        
        for xml_file in annotation_dir.glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            filename = root.find('filename').text
            image_path = image_dir / filename
            
            if image_path.exists():
                self.images.append(str(image_path))
            
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')
                
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                self.annotations.append({
                    'image': xml_file.stem,
                    'class': class_name,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'width': width,
                    'height': height,
                })
    
    def convert(self, source_format: str, target_format: str, output_dir: str):
        """
        Convert between formats
        
        Args:
            source_format: Source format
            target_format: Target format
            output_dir: Output directory
        
        Returns:
            Path to converted dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if target_format == 'yolo':
            return self._convert_to_yolo(output_path)
        elif target_format == 'coco':
            return self._convert_to_coco(output_path)
        elif target_format == 'voc':
            return self._convert_to_voc(output_path)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    def _convert_to_yolo(self, output_dir: Path) -> Path:
        """Convert to YOLO format"""
        labels_dir = output_dir / 'labels'
        images_dir = output_dir / 'images'
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Group annotations by image
        annotations_by_image = defaultdict(list)
        for ann in self.annotations:
            annotations_by_image[ann['image']].append(ann)
        
        for image_name, anns in annotations_by_image.items():
            label_file = labels_dir / f"{image_name}.txt"
            
            with open(label_file, 'w') as f:
                for ann in anns:
                    if self.format == 'yolo':
                        # Already in YOLO format
                        f.write(f"{ann['class']} {ann['x_center']} {ann['y_center']} "
                               f"{ann['width']} {ann['height']}\n")
                    elif self.format == 'voc':
                        # Convert VOC to YOLO
                        img_width = ann['width']
                        img_height = ann['height']
                        x_center = ((ann['xmin'] + ann['xmax']) / 2) / img_width
                        y_center = ((ann['ymin'] + ann['ymax']) / 2) / img_height
                        width = (ann['xmax'] - ann['xmin']) / img_width
                        height = (ann['ymax'] - ann['ymin']) / img_height
                        
                        # Get class index
                        class_idx = self._get_class_index(ann['class'])
                        f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
        
        return output_dir
    
    def _convert_to_coco(self, output_dir: Path) -> Path:
        """Convert to COCO format"""
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Create categories
        unique_classes = set()
        for ann in self.annotations:
            if isinstance(ann['class'], str):
                unique_classes.add(ann['class'])
            else:
                unique_classes.add(ann['class'])
        
        for idx, cls in enumerate(sorted(unique_classes)):
            coco_data['categories'].append({
                'id': idx,
                'name': str(cls),
                'supercategory': 'none'
            })
        
        # Add images and annotations
        image_id_map = {}
        for idx, img in enumerate(self.images):
            image_id_map[Path(img).stem] = idx
            coco_data['images'].append({
                'id': idx,
                'file_name': Path(img).name,
                'width': 640,  # Default, should be read from actual image
                'height': 640,
            })
        
        for idx, ann in enumerate(self.annotations):
            image_id = image_id_map.get(ann['image'], 0)
            coco_data['annotations'].append({
                'id': idx,
                'image_id': image_id,
                'category_id': ann['class'],
                'bbox': [0, 0, 0, 0],  # Should be calculated from actual coords
                'area': 0,
                'iscrowd': 0,
            })
        
        # Save JSON
        output_file = output_dir / 'annotations.json'
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        return output_dir
    
    def _convert_to_voc(self, output_dir: Path) -> Path:
        """Convert to Pascal VOC format"""
        annotations_dir = output_dir / 'Annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by image and create XML files
        annotations_by_image = defaultdict(list)
        for ann in self.annotations:
            annotations_by_image[ann['image']].append(ann)
        
        for image_name, anns in annotations_by_image.items():
            # Create VOC XML structure
            root = ET.Element('annotation')
            ET.SubElement(root, 'filename').text = f"{image_name}.jpg"
            
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = '640'
            ET.SubElement(size, 'height').text = '640'
            ET.SubElement(size, 'depth').text = '3'
            
            for ann in anns:
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = str(ann['class'])
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'
                
                bndbox = ET.SubElement(obj, 'bndbox')
                # These should be calculated from actual annotations
                ET.SubElement(bndbox, 'xmin').text = '0'
                ET.SubElement(bndbox, 'ymin').text = '0'
                ET.SubElement(bndbox, 'xmax').text = '100'
                ET.SubElement(bndbox, 'ymax').text = '100'
            
            tree = ET.ElementTree(root)
            xml_file = annotations_dir / f"{image_name}.xml"
            tree.write(xml_file, encoding='utf-8', xml_declaration=True)
        
        return output_dir
    
    def _get_class_index(self, class_name: Union[str, int]) -> int:
        """Get class index from name"""
        if isinstance(class_name, int):
            return class_name
        
        # Try to find in categories
        for idx, cat in enumerate(self.categories):
            if cat.get('name') == class_name:
                return idx
        
        return 0  # Default
    
    def validate_annotations(self) -> Dict[str, Any]:
        """
        Validate annotations
        
        Returns:
            Validation report
        """
        report = {
            'total_images': len(self.images),
            'total_annotations': len(self.annotations),
            'issues': []
        }
        
        # Check for images without annotations
        annotated_images = set(ann['image'] for ann in self.annotations)
        all_images = set(Path(img).stem for img in self.images)
        
        images_without_annotations = all_images - annotated_images
        if images_without_annotations:
            report['issues'].append({
                'type': 'missing_annotations',
                'count': len(images_without_annotations),
                'samples': list(images_without_annotations)[:5]
            })
        
        # Check for valid bounding boxes
        invalid_boxes = []
        for ann in self.annotations:
            if self.format == 'yolo':
                if not (0 <= ann['x_center'] <= 1 and 0 <= ann['y_center'] <= 1):
                    invalid_boxes.append(ann)
                if not (0 < ann['width'] <= 1 and 0 < ann['height'] <= 1):
                    invalid_boxes.append(ann)
        
        if invalid_boxes:
            report['issues'].append({
                'type': 'invalid_boxes',
                'count': len(invalid_boxes)
            })
        
        return report
    
    def create_data_yaml(
        self,
        output_path: str,
        class_names: List[str],
        train_path: str = 'train',
        val_path: str = 'val',
        test_path: str = None
    ):
        """
        Generate data.yaml for YOLO training
        
        Args:
            output_path: Path to save data.yaml
            class_names: List of class names
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Optional path to test data
        """
        data_config = {
            'path': str(self.root_dir),
            'train': train_path,
            'val': val_path,
            'nc': len(class_names),
            'names': class_names
        }
        
        if test_path:
            data_config['test'] = test_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    def split_dataset(
        self,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Split dataset into train/val/test sets
        
        Args:
            output_dir: Output directory
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
        """
        import random
        random.seed(random_seed)
        
        output_path = Path(output_dir)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Shuffle images
        images = list(self.images)
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy files to respective directories
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split_name, split_images in splits.items():
            for img_path in split_images:
                img_path = Path(img_path)
                # Copy image
                shutil.copy(img_path, output_path / split_name / 'images' / img_path.name)
                
                # Copy corresponding label if exists
                label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy(
                        label_path,
                        output_path / split_name / 'labels' / f"{img_path.stem}.txt"
                    )
        
        return output_path
