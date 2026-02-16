#!/usr/bin/env python3
"""Validate project structure and files."""
import os
import sys
from pathlib import Path


def check_file_exists(path, description):
    """Check if file exists."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} missing: {path}")
        return False


def check_directory_exists(path, description):
    """Check if directory exists."""
    if Path(path).is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} missing: {path}")
        return False


def main():
    """Main validation function."""
    print("=" * 70)
    print("Project Structure Validation")
    print("=" * 70)
    
    checks = []
    
    # Configuration files
    print("\n📋 Configuration Files:")
    checks.append(check_file_exists("configs/data_config.yaml", "Data config"))
    checks.append(check_file_exists("configs/model_config.yaml", "Model config"))
    checks.append(check_file_exists("configs/training_config.yaml", "Training config"))
    checks.append(check_file_exists("configs/deployment_config.yaml", "Deployment config"))
    
    # Core source files
    print("\n🔧 Core Source Files:")
    checks.append(check_file_exists("src/data/adapters/csv_adapter.py", "CSV Adapter"))
    checks.append(check_file_exists("src/data/adapters/image_adapter.py", "Image Adapter"))
    checks.append(check_file_exists("src/data/preprocessors/tabular_preprocessor.py", "Tabular Preprocessor"))
    checks.append(check_file_exists("src/data/preprocessors/image_preprocessor.py", "Image Preprocessor"))
    checks.append(check_file_exists("src/models/base_model.py", "Base Model"))
    checks.append(check_file_exists("src/models/tabular_model.py", "Tabular Model"))
    checks.append(check_file_exists("src/models/vision_model.py", "Vision Model"))
    checks.append(check_file_exists("src/training/trainer.py", "Trainer"))
    checks.append(check_file_exists("src/training/callbacks.py", "Callbacks"))
    checks.append(check_file_exists("src/training/metrics.py", "Metrics"))
    checks.append(check_file_exists("src/inference/predictor.py", "Predictor"))
    checks.append(check_file_exists("src/inference/serving.py", "FastAPI Serving"))
    
    # Scripts
    print("\n📜 Entry Point Scripts:")
    checks.append(check_file_exists("scripts/train.py", "Training Script"))
    checks.append(check_file_exists("scripts/evaluate.py", "Evaluation Script"))
    checks.append(check_file_exists("scripts/predict.py", "Prediction Script"))
    checks.append(check_file_exists("scripts/serve.py", "Serving Script"))
    checks.append(check_file_exists("scripts/setup_data.py", "Setup Data Script"))
    
    # Tests
    print("\n🧪 Test Files:")
    checks.append(check_file_exists("tests/conftest.py", "Pytest Config"))
    checks.append(check_file_exists("tests/unit/test_config_loader.py", "Config Loader Tests"))
    checks.append(check_file_exists("tests/unit/test_csv_adapter.py", "CSV Adapter Tests"))
    checks.append(check_file_exists("tests/unit/test_tabular_model.py", "Tabular Model Tests"))
    checks.append(check_file_exists("tests/unit/test_metrics.py", "Metrics Tests"))
    
    # CI/CD
    print("\n🔄 CI/CD Workflows:")
    checks.append(check_file_exists(".github/workflows/ci.yaml", "CI Workflow"))
    checks.append(check_file_exists(".github/workflows/train.yaml", "Training Workflow"))
    
    # Documentation
    print("\n📚 Documentation:")
    checks.append(check_file_exists("README.md", "README"))
    checks.append(check_file_exists("docs/SPEC.md", "Specification"))
    checks.append(check_file_exists("docs/ARCHITECTURE.md", "Architecture"))
    checks.append(check_file_exists("docs/API.md", "API Documentation"))
    
    # Build files
    print("\n🏗️ Build Files:")
    checks.append(check_file_exists("requirements.txt", "Requirements"))
    checks.append(check_file_exists("setup.py", "Setup"))
    checks.append(check_file_exists("Makefile", "Makefile"))
    checks.append(check_file_exists("Dockerfile", "Dockerfile"))
    checks.append(check_file_exists(".gitignore", "Gitignore"))
    
    # Directories
    print("\n📁 Required Directories:")
    checks.append(check_directory_exists("artifacts/models", "Models Directory"))
    checks.append(check_directory_exists("artifacts/logs", "Logs Directory"))
    checks.append(check_directory_exists("artifacts/metrics", "Metrics Directory"))
    checks.append(check_directory_exists("data/sample", "Sample Data Directory"))
    
    # Summary
    print("\n" + "=" * 70)
    passed = sum(checks)
    total = len(checks)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print(f"Validation Results: {passed}/{total} checks passed ({percentage:.1f}%)")
    
    if passed == total:
        print("✅ All checks passed! Project structure is complete.")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
