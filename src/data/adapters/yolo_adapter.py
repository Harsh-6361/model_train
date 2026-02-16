"""YOLO data adapter for object detection tasks."""
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .base_adapter import BaseDataAdapter


class YOLOAdapter(BaseDataAdapter):
    """Adapter for YOLO format object detection data."""
    
    def load(self, path: str, **kwargs) -> Dict[str, Any]:
        """
        Load YOLO dataset.
        
        Args:
            path: Path to dataset directory
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing dataset information
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"YOLO directory not found: {path}")
        
        # Check for data.yaml
        data_yaml = path / "data.yaml"
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            return data_config
        
        # Otherwise, scan directory structure
        dataset_info = {
            'path': str(path),
            'train': str(path / 'train'),
            'val': str(path / 'val'),
            'test': str(path / 'test'),
            'names': [],
            'nc': 0
        }
        
        return dataset_info
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate YOLO dataset.
        
        Args:
            data: Dataset configuration
            
        Returns:
            True if valid
        """
        required_keys = ['path', 'train', 'val', 'names', 'nc']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key in YOLO config: {key}")
        
        # Check that paths exist
        base_path = Path(data['path'])
        if not base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")
        
        return True
    
    def preprocess(self, data: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Preprocess YOLO dataset.
        
        Args:
            data: Dataset configuration
            config: Preprocessing configuration
            
        Returns:
            Processed dataset configuration
        """
        # No preprocessing needed for YOLO format
        return data
    
    def convert_coco_to_yolo(
        self,
        coco_json: str,
        output_dir: str,
        image_dir: str
    ) -> None:
        """
        Convert COCO format to YOLO format.
        
        Args:
            coco_json: Path to COCO JSON file
            output_dir: Output directory for YOLO format
            image_dir: Directory containing images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
        
        # Create category mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        cat_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
        
        # Process annotations
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Create YOLO txt files
        for img in coco_data['images']:
            img_id = img['id']
            img_w = img['width']
            img_h = img['height']
            
            # Create txt file
            txt_path = output_dir / f"{Path(img['file_name']).stem}.txt"
            
            with open(txt_path, 'w') as f:
                if img_id in img_to_anns:
                    for ann in img_to_anns[img_id]:
                        # Convert bbox from [x, y, w, h] to [x_center, y_center, w, h] normalized
                        x, y, w, h = ann['bbox']
                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        w_norm = w / img_w
                        h_norm = h / img_h
                        
                        class_id = cat_id_to_yolo[ann['category_id']]
                        f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")
        
        # Create data.yaml
        data_yaml = {
            'path': str(output_dir.parent),
            'train': str(output_dir / 'train'),
            'val': str(output_dir / 'val'),
            'names': [categories[cat_id] for cat_id in sorted(categories.keys())],
            'nc': len(categories)
        }
        
        with open(output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
    
    def convert_voc_to_yolo(
        self,
        voc_dir: str,
        output_dir: str
    ) -> None:
        """
        Convert Pascal VOC format to YOLO format.
        
        Args:
            voc_dir: Directory containing VOC XML files
            output_dir: Output directory for YOLO format
        """
        voc_dir = Path(voc_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all classes
        classes = set()
        xml_files = list(voc_dir.glob('*.xml'))
        
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                classes.add(obj.find('name').text)
        
        classes = sorted(list(classes))
        class_to_id = {cls: idx for idx, cls in enumerate(classes)}
        
        # Convert each XML file
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            
            txt_path = output_dir / f"{xml_file.stem}.txt"
            
            with open(txt_path, 'w') as f:
                for obj in root.findall('object'):
                    cls = obj.find('name').text
                    class_id = class_to_id[cls]
                    
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # Convert to YOLO format
                    x_center = ((xmin + xmax) / 2) / img_w
                    y_center = ((ymin + ymax) / 2) / img_h
                    w = (xmax - xmin) / img_w
                    h = (ymax - ymin) / img_h
                    
                    f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")
        
        # Create data.yaml
        data_yaml = {
            'path': str(output_dir.parent),
            'train': str(output_dir / 'train'),
            'val': str(output_dir / 'val'),
            'names': classes,
            'nc': len(classes)
        }
        
        with open(output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
    
    def create_data_yaml(
        self,
        data_dir: str,
        class_names: List[str],
        train_path: str = 'train',
        val_path: str = 'val',
        test_path: Optional[str] = 'test'
    ) -> str:
        """
        Create data.yaml file for YOLO training.
        
        Args:
            data_dir: Base directory for dataset
            class_names: List of class names
            train_path: Relative path to training data
            val_path: Relative path to validation data
            test_path: Optional relative path to test data
            
        Returns:
            Path to created data.yaml file
        """
        data_dir = Path(data_dir)
        
        data_yaml = {
            'path': str(data_dir),
            'train': train_path,
            'val': val_path,
            'names': class_names,
            'nc': len(class_names)
        }
        
        if test_path:
            data_yaml['test'] = test_path
        
        yaml_path = data_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        return str(yaml_path)
