# Architecture Documentation

## System Overview

The Multi-Modal ML Training System is designed with a modular, layered architecture that separates concerns and enables extensibility. The system supports both tabular and vision models with a unified interface.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  (CLI Scripts, API Clients, External Applications)              │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                      Interface Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   train.py   │  │  serve.py    │  │ predict.py   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                      Application Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Trainer    │  │  Predictor   │  │FastAPI Server│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                        Core Layers                               │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   Model Layer                        │        │
│  │  BaseModel → TabularModel → MLPClassifier           │        │
│  │           → VisionModel → ResNet/EfficientNet       │        │
│  │  ModelRegistry (versioning & metadata)              │        │
│  └─────────────────────────────────────────────────────┘        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   Data Layer                         │        │
│  │  Adapters: CSVAdapter, ImageAdapter                 │        │
│  │  Validators: SchemaValidator                         │        │
│  │  Preprocessors: TabularPreprocessor, ImagePre...    │        │
│  │  Loaders: UnifiedDataLoader (PyTorch)               │        │
│  └─────────────────────────────────────────────────────┘        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                 Training Layer                       │        │
│  │  Trainer, Callbacks, Metrics                        │        │
│  └─────────────────────────────────────────────────────┘        │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                   Cross-Cutting Concerns                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Observability│  │    Utils     │  │   Config     │          │
│  │ (Logging,    │  │  (Helpers,   │  │  (YAML/JSON  │          │
│  │  Metrics,    │  │   Seed)      │  │   Loader)    │          │
│  │  Health)     │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Data Storage │  │Model Artifacts│ │Logs & Metrics│          │
│  │  (CSV, Img)  │  │   (.pth)      │ │   (JSON)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### 1. Data Layer

#### Adapters
**Purpose**: Abstract data loading from different sources

- **CSVAdapter**: Loads tabular data from CSV files
  - Handles train/val/test splitting
  - Supports schema validation
  - Provides data information

- **ImageAdapter**: Loads images from directory structure
  - Expects class-based subdirectories
  - Handles class mapping
  - Supports stratified splitting

#### Validators
**Purpose**: Ensure data quality and schema compliance

- **SchemaValidator**: Validates data against JSON schemas
  - Checks required columns/formats
  - Validates data types
  - Checks for null values

#### Preprocessors
**Purpose**: Transform raw data for model consumption

- **TabularPreprocessor**: Handles structured data
  - Normalization (StandardScaler)
  - Missing value imputation
  - Categorical encoding (OneHot/Label)

- **ImagePreprocessor**: Handles image data
  - Resizing and normalization
  - Data augmentation (flip, rotate, color jitter)
  - Separate train/val transforms

#### Loaders
**Purpose**: Create PyTorch-compatible data loaders

- **UnifiedDataLoader**: Factory for creating DataLoader instances
  - TabularDataset (numpy → torch tensors)
  - ImageDataset (PIL → torch tensors)
  - Configurable batch size and workers

### 2. Model Layer

#### Base Model
**Purpose**: Define common model interface

```python
class BaseModel(ABC):
    - forward(): Abstract method for inference
    - save/load(): Model persistence
    - get_config(): Configuration retrieval
    - count_parameters(): Model size
    - freeze/unfreeze(): Fine-tuning support
```

#### Model Implementations

**TabularModel**
- MLPClassifier: Multi-layer perceptron
  - Configurable hidden layers
  - Dropout regularization
  - Multiple activation functions

**VisionModel**
- ResNetClassifier: ResNet-based (18/34/50)
  - Pretrained backbone support
  - Custom classification head
  - Backbone freezing capability

- EfficientNetClassifier: EfficientNet-based (B0/B1)
  - State-of-the-art efficiency
  - Compound scaling
  - Pretrained weights

#### Model Registry
**Purpose**: Track model versions and metadata

- Stores model metadata (metrics, config)
- Supports querying (best, latest)
- Enables model comparison
- JSON-based storage

### 3. Training Layer

#### Trainer
**Purpose**: Unified training orchestration

- Single interface for all model types
- Epoch and batch iteration
- Training and validation loops
- Metrics calculation
- Callback management

#### Callbacks
**Purpose**: Extend training behavior

- **EarlyStopping**: Stop training when metric stops improving
- **ModelCheckpoint**: Save best/last model
- **MetricsLogger**: Log metrics to file

#### Metrics
**Purpose**: Evaluate model performance

- MetricsCalculator: Sklearn-based metrics
  - Accuracy, Precision, Recall, F1
  - Confusion matrix
  - AUC (if probabilities available)

- MetricsTracker: Track metrics over epochs
  - History storage
  - Best/latest retrieval

### 4. Inference Layer

#### Predictor
**Purpose**: Run predictions on new data

- Loads trained models
- Applies preprocessing
- Supports batch prediction
- Returns formatted results with labels

#### Serving
**Purpose**: Expose models via REST API

- FastAPI-based server
- Health check endpoint
- Metrics endpoint
- Tabular and image prediction endpoints
- Request/response validation

### 5. Observability Layer

#### Logger
**Purpose**: Structured logging

- JSON-formatted logs
- Contextual information
- Multiple log levels
- File and console output

#### Metrics Collector
**Purpose**: Track operational metrics

- Training metrics (loss, accuracy)
- Inference latency
- Request counts
- Prometheus-compatible

#### Health Check
**Purpose**: Monitor system health

- System resources (CPU, memory, disk)
- Model availability
- GPU availability (if applicable)
- Overall status

### 6. Utilities

#### Config Loader
- YAML/JSON parsing
- Configuration merging
- Deep merge support

#### Helpers
- Seed setting (reproducibility)
- Device selection (CPU/CUDA/MPS)
- Directory creation
- Time formatting

## Data Flow

### Training Flow

```
CSV/Images → Adapter → Validator → Preprocessor → Loader
    ↓
  Model ← Trainer ← Callbacks
    ↓
Checkpoint → Registry
```

### Inference Flow

```
Input → Preprocessor → Model → Predictor → API → Response
                                    ↓
                                 Metrics
```

## Configuration Management

### Configuration Files
- `data_config.yaml`: Data paths and preprocessing
- `model_config.yaml`: Model architecture
- `training_config.yaml`: Training parameters
- `deployment_config.yaml`: Serving configuration

### Configuration Hierarchy
1. Default values in code
2. Configuration files
3. Command-line arguments (override files)

## Design Patterns

### 1. Abstract Factory
- TabularModel.create()
- VisionModel.create()
- UnifiedDataLoader factory methods

### 2. Template Method
- BaseModel defines interface
- Subclasses implement specifics

### 3. Strategy
- Different preprocessing strategies
- Different optimizer strategies

### 4. Observer
- Callback pattern for training events

### 5. Builder
- Trainer configuration builder

## Extension Points

### Adding New Model Types
1. Subclass BaseModel
2. Implement forward() and get_config()
3. Add to model factory

### Adding New Data Sources
1. Create new Adapter class
2. Implement load() and split_data()
3. Add to data loading logic

### Adding New Metrics
1. Add method to MetricsCalculator
2. Include in metric_names config

### Adding New Callbacks
1. Subclass Callback
2. Implement relevant methods
3. Add to trainer callbacks list

## Scalability Considerations

### Current Limitations
- Single-node training
- CPU/Single GPU
- In-memory data processing

### Future Scalability
- Distributed training (DDP, Ray)
- Multi-GPU support (DataParallel)
- Data streaming (for large datasets)
- Model parallelism (for large models)
- Horizontal scaling (API replicas)

## Security Considerations

- Input validation (schemas, Pydantic)
- No hardcoded credentials
- Configurable access control (future)
- Model integrity checks
- Secure model storage

## Performance Optimization

### Current
- PyTorch tensor operations (GPU-accelerated)
- Data loader workers (parallel loading)
- Batch prediction support

### Future
- Model quantization
- ONNX export
- TensorRT optimization
- Caching strategies

## Monitoring and Debugging

### Logging Levels
- DEBUG: Detailed internal state
- INFO: General progress
- WARNING: Non-critical issues
- ERROR: Critical failures

### Metrics
- Training: loss, accuracy, epoch time
- Inference: latency, throughput
- System: CPU, memory, GPU utilization

### Health Checks
- Liveness: Is service running?
- Readiness: Can service handle requests?
- Model: Is model loaded and functional?
