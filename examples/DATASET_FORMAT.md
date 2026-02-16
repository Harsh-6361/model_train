# Example Dataset Structure

This document describes the expected dataset structure for different annotation formats.

## YOLO Format (Recommended)

```
data/
├── raw/
│   ├── images/
│   │   ├── train/
│   │   │   ├── img001.jpg
│   │   │   ├── img002.jpg
│   │   │   └── ...
│   │   ├── val/
│   │   │   ├── img101.jpg
│   │   │   └── ...
│   │   └── test/
│   │       ├── img201.jpg
│   │       └── ...
│   └── labels/
│       ├── train/
│       │   ├── img001.txt
│       │   ├── img002.txt
│       │   └── ...
│       ├── val/
│       │   ├── img101.txt
│       │   └── ...
│       └── test/
│           ├── img201.txt
│           └── ...
└── processed/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

### Label File Format (YOLO)

Each `.txt` file contains one line per object:

```
class_id x_center y_center width height
```

Where:
- `class_id`: Integer class ID (0-indexed)
- `x_center`: X coordinate of box center (normalized 0-1)
- `y_center`: Y coordinate of box center (normalized 0-1)
- `width`: Box width (normalized 0-1)
- `height`: Box height (normalized 0-1)

**Example** (`img001.txt`):
```
0 0.716797 0.395833 0.216406 0.147222
1 0.687500 0.379167 0.255469 0.158333
2 0.420312 0.395833 0.140625 0.166667
```

### data.yaml Format

```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images

nc: 3  # number of classes
names: ['class1', 'class2', 'class3']  # class names
```

## COCO Format

```
data/
├── raw/
│   ├── images/
│   │   ├── train2017/
│   │   ├── val2017/
│   │   └── test2017/
│   └── annotations/
│       ├── instances_train2017.json
│       ├── instances_val2017.json
│       └── instances_test2017.json
```

### Annotation JSON Format (COCO)

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img001.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 50, 75],
      "area": 3750,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "none"
    }
  ]
}
```

## Pascal VOC Format

```
data/
├── raw/
│   ├── JPEGImages/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── Annotations/
│   │   ├── img001.xml
│   │   ├── img002.xml
│   │   └── ...
│   └── ImageSets/
│       └── Main/
│           ├── train.txt
│           ├── val.txt
│           └── test.txt
```

### Annotation XML Format (VOC)

```xml
<annotation>
  <filename>img001.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>150</xmax>
      <ymax>275</ymax>
    </bndbox>
  </object>
</annotation>
```

## Converting Between Formats

Use the data preparation script to convert:

```bash
# COCO to YOLO
python scripts/prepare_data.py \
  --input data/raw/coco/ \
  --format coco \
  --output data/processed/ \
  --split

# VOC to YOLO
python scripts/prepare_data.py \
  --input data/raw/voc/ \
  --format voc \
  --output data/processed/ \
  --split
```

## Sample Dataset

For testing, you can download sample datasets:

### COCO Sample
```bash
# Download COCO val2017 (1GB)
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

### Pascal VOC Sample
```bash
# Download VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

## Class Configuration

Update `configs/data_config.yaml` with your classes:

```yaml
data:
  classes:
    - "person"
    - "car"
    - "dog"
    - "cat"
    # Add all your classes here
```

## Validation

After preparing data, validate it:

```bash
python scripts/validate_data.py --config configs/data_config.yaml
```

Expected output:
```
✓ Found data.yaml: data/processed/data.yaml
  Number of classes: 3
  Classes: ['person', 'car', 'dog']

✓ Successfully loaded dataset
  Images: 1000
  Annotations: 5432

✓ Train split:
  Images: 800
  Labels: 800

✓ Val split:
  Images: 100
  Labels: 100

✓ Test split:
  Images: 100
  Labels: 100

✓ No issues found! Dataset is valid.
```
