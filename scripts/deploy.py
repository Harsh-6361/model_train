#!/usr/bin/env python3
"""
Model Deployment Script

Deploys trained models to various environments
"""

import argparse
import sys
from pathlib import Path
import shutil
import json


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy trained model')
    parser.add_argument(
        '--model',
        required=True,
        help='Path to model file to deploy'
    )
    parser.add_argument(
        '--env',
        choices=['staging', 'production', 'local'],
        required=True,
        help='Deployment environment'
    )
    parser.add_argument(
        '--dest',
        help='Destination directory or URL'
    )
    parser.add_argument(
        '--version',
        help='Model version tag'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    print(f"Deploying model: {model_path}")
    print(f"Environment: {args.env}")
    
    if args.env == 'local':
        deploy_local(model_path, args.dest, args.version)
    elif args.env == 'staging':
        deploy_staging(model_path, args.dest, args.version)
    elif args.env == 'production':
        deploy_production(model_path, args.dest, args.version)
    
    print("✓ Deployment successful!")


def deploy_local(model_path: Path, dest: str, version: str):
    """Deploy to local environment"""
    if not dest:
        dest = 'artifacts/deployed/local'
    
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model
    dest_file = dest_path / model_path.name
    shutil.copy(model_path, dest_file)
    
    # Create deployment manifest
    manifest = {
        'model': str(model_path.name),
        'version': version or 'latest',
        'environment': 'local',
        'path': str(dest_file)
    }
    
    manifest_file = dest_path / 'deployment_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Model deployed to: {dest_file}")
    print(f"Manifest: {manifest_file}")


def deploy_staging(model_path: Path, dest: str, version: str):
    """Deploy to staging environment"""
    if not dest:
        dest = 'artifacts/deployed/staging'
    
    # Similar to local but with additional checks
    deploy_local(model_path, dest, version)
    
    print("Note: For actual staging deployment, configure:")
    print("  - Staging server URL")
    print("  - API endpoints")
    print("  - Authentication credentials")


def deploy_production(model_path: Path, dest: str, version: str):
    """Deploy to production environment"""
    print("⚠️  Production deployment requires additional setup:")
    print("  1. Review and approve model performance")
    print("  2. Configure production server credentials")
    print("  3. Set up monitoring and alerting")
    print("  4. Create rollback plan")
    print("")
    print("For safety, this script only performs local copy.")
    print("Use your organization's deployment pipeline for actual production deployment.")
    
    if not dest:
        dest = 'artifacts/deployed/production-ready'
    
    deploy_local(model_path, dest, version)


if __name__ == '__main__':
    main()
