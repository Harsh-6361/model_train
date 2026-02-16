# System Architecture

## Overview

The ML Pipeline system is designed as a modular, scalable platform for training and deploying multi-modal machine learning models. The architecture follows clean architecture principles with clear separation of concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  (CLI Scripts, API Endpoints, CI/CD Workflows)              │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐│
│  │ Training │  │Inference │  │Evaluation│  │Observability││
│  │ Pipeline │  │  Engine  │  │ Pipeline │  │   Layer     ││
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘│
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                      Core Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐│
│  │  Data    │  │  Models  │  │ Training │  │Observability││
│  │ Adapters │  │ Registry │  │Callbacks │  │  Metrics    ││
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘│
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐│
│  │PyTorch   │  │ FastAPI  │  │MLflow    │  │ Prometheus  ││
│  │Framework │  │ Server   │  │Tracking  │  │   Metrics   ││
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Data Layer (`src/data/`)

**Purpose**: Handle all data ingestion, validation, preprocessing, and loading.

**Components**:
- **Adapters**: Interface to different data sources (CSV, Images, YOLO)
  - `BaseAdapter`: Abstract base class defining the adapter interface
  - `CSVAdapter`: Handles tabular data from CSV files
  - `ImageAdapter`: Handles image data from directories
  - `YOLOAdapter`: Handles YOLO format detection data
  
- **Validators**: Ensure data quality and schema compliance
  - `SchemaValidator`: Validates data against JSON schemas
  
- **Preprocessors**: Transform raw data into model-ready format
  - `TabularPreprocessor`: Scaling, encoding, imputation
  - `ImagePreprocessor`: Resizing, augmentation, normalization
  
- **Loaders**: Create PyTorch DataLoaders
  - `UnifiedLoader`: Single interface for all model types
  - `LargeScaleLoader`: Optimized for distributed training

**Design Patterns**:
- Adapter Pattern for data sources
- Strategy Pattern for preprocessing
- Factory Pattern for loader creation

### 2. Model Layer (`src/models/`)

**Purpose**: Define and manage model architectures.

**Components**:
- `BaseModel`: Abstract base class for all models
  - Defines interface: forward, training_step, validation_step
  - Provides save/load functionality
  - Parameter counting utilities
  
- `TabularModel` (MLP): Multi-layer perceptron for structured data
  - Configurable architecture
  - Support for classification and regression
  
- `VisionModel`: CNN-based models for image classification
  - Pretrained backbones (ResNet, EfficientNet, MobileNet)
  - Transfer learning support
  
- `YOLOModel`: Object detection models
  - YOLOv5/v8 integration via ultralytics
  - Export to ONNX, TorchScript
  
- `ModelRegistry`: Factory for model creation
  - Centralized model instantiation
  - Configuration-driven model selection

**Design Patterns**:
- Template Method Pattern (BaseModel)
- Factory Pattern (ModelRegistry)
- Adapter Pattern (YOLOModel wraps ultralytics)

### 3. Training Layer (`src/training/`)

**Purpose**: Orchestrate model training with callbacks and metrics.

**Components**:
- `Trainer`: Main training orchestrator
  - Handles training loop
  - Manages callbacks
  - Optimizer and scheduler management
  - Mixed precision training support
  
- `DistributedTrainer`: Extension for multi-GPU training
  - Wraps model in DistributedDataParallel
  - Process group management
  
- `Callbacks`: Extensible callback system
  - `EarlyStopping`: Stop training when no improvement
  - `ModelCheckpoint`: Save best/last models
  - `LearningRateScheduler`: Adjust learning rate
  - `MetricsLogger`: Log metrics to file
  - `ProgressPrinter`: Display training progress
  
- `Metrics`: Calculate and track metrics
  - Classification metrics (accuracy, F1, precision, recall)
  - Regression metrics (MSE, MAE, R²)
  - AverageMeter for tracking running averages

**Design Patterns**:
- Observer Pattern (callbacks)
- Strategy Pattern (optimizer/scheduler selection)
- Template Method Pattern (training loop)

### 4. Inference Layer (`src/inference/`)

**Purpose**: Run predictions and serve models via API.

**Components**:
- `Predictor`: Inference engine
  - Load and run models
  - Handle different input types
  - Batch inference support
  
- `FastAPI Server`: REST API for predictions
  - Health check endpoint
  - Prediction endpoints (tabular, image)
  - Metrics endpoint (Prometheus)
  - Async request handling
  - CORS support

**Design Patterns**:
- Facade Pattern (Predictor simplifies inference)
- Singleton Pattern (global predictor instance)

### 5. Observability Layer (`src/observability/`)

**Purpose**: Monitor, log, and track experiments.

**Components**:
- `Logger`: Structured logging with JSON output
  - Console and file handlers
  - Configurable log levels
  - Context injection
  
- `MetricsCollector`: Prometheus metrics
  - Training metrics (loss, accuracy)
  - Inference metrics (latency, throughput)
  - System metrics (GPU, memory)
  
- `ExperimentTracker`: MLflow integration
  - Parameter logging
  - Metrics tracking
  - Artifact management
  - Model registry
  
- `HealthCheck`: System health monitoring
  - PyTorch availability
  - CUDA/GPU status
  - Model loading status

**Design Patterns**:
- Singleton Pattern (global instances)
- Observer Pattern (metrics collection)

## Data Flow

### Training Flow

```
1. Configuration Loading
   ├─ Load data_config.yaml
   ├─ Load model_config.yaml
   └─ Load training_config.yaml
   
2. Data Preparation
   ├─ Adapter.load() → Raw data
   ├─ Adapter.validate() → Validated data
   ├─ Adapter.preprocess() → Preprocessed data
   ├─ Adapter.split() → Train/Val/Test splits
   └─ Loader.get_loaders() → PyTorch DataLoaders
   
3. Model Creation
   ├─ Registry.create() → Model instance
   └─ Model.to(device) → Move to GPU/CPU
   
4. Training Loop
   ├─ For each epoch:
   │  ├─ Training phase
   │  │  ├─ For each batch:
   │  │  │  ├─ model.training_step()
   │  │  │  ├─ loss.backward()
   │  │  │  └─ optimizer.step()
   │  │  └─ Track metrics
   │  ├─ Validation phase
   │  │  ├─ For each batch:
   │  │  │  └─ model.validation_step()
   │  │  └─ Track metrics
   │  └─ Callbacks (checkpoint, early stopping, etc.)
   └─ Save final model
   
5. Evaluation
   ├─ Load test data
   ├─ model.test_step() for each batch
   └─ Calculate final metrics
```

### Inference Flow

```
1. API Request
   ├─ POST /predict or /predict/image
   └─ Parse request body/file
   
2. Data Preprocessing
   ├─ Convert to numpy/tensor
   ├─ Apply transformations
   └─ Move to device
   
3. Prediction
   ├─ model.eval()
   ├─ with torch.no_grad():
   │  └─ output = model(input)
   └─ Post-process output
   
4. Response
   ├─ Convert to JSON
   ├─ Record metrics
   └─ Return to client
```

## Scalability Considerations

### Horizontal Scaling
- Multiple API server instances behind load balancer
- Distributed training across multiple GPUs/nodes
- Asynchronous inference queue for batch processing

### Vertical Scaling
- Mixed precision training (FP16)
- Gradient accumulation for larger effective batch sizes
- Gradient checkpointing for memory efficiency
- Model quantization for faster inference

### Data Scaling
- Streaming data loading from cloud storage
- Distributed data preprocessing
- Efficient data caching and prefetching

## Security Considerations

- Input validation for all API endpoints
- Rate limiting to prevent abuse
- Model file integrity checks
- Secure configuration management (no secrets in code)
- CORS configuration for API access control

## Deployment Architecture

### Local Development
```
Developer Machine
├─ Python environment
├─ Local GPU (optional)
└─ Local file storage
```

### Docker Deployment
```
Docker Container
├─ Application code
├─ Dependencies
├─ Model artifacts (volume mount)
└─ Data (volume mount)
```

### Multi-Service Deployment
```
Docker Compose / Kubernetes
├─ API Service (FastAPI)
├─ Prometheus (metrics)
├─ Grafana (visualization)
└─ MLflow (experiment tracking)
```

## Technology Stack

- **Framework**: PyTorch 2.0+
- **API**: FastAPI
- **Validation**: Pydantic
- **Logging**: structlog
- **Metrics**: Prometheus
- **Experiment Tracking**: MLflow
- **Testing**: pytest
- **Code Quality**: black, isort, flake8
- **CI/CD**: GitHub Actions
- **Containerization**: Docker
- **Orchestration**: Docker Compose / Kubernetes

## Future Enhancements

1. **Advanced Models**
   - Transformer models for NLP tasks
   - Graph Neural Networks
   - AutoML capabilities

2. **Advanced Features**
   - Model versioning and A/B testing
   - Automated hyperparameter tuning
   - Neural architecture search
   - Model interpretability tools

3. **Infrastructure**
   - Kubernetes deployment
   - Cloud provider integration (AWS, GCP, Azure)
   - Advanced monitoring dashboards
   - Automated model retraining pipelines
