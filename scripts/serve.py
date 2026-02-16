#!/usr/bin/env python3
"""
API server entrypoint.
Usage: python scripts/serve.py --port 8000 --model artifacts/models/best.pt --model-type tabular
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.inference.serving import create_app
from src.utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description='Start API server')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to run server on')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to run server on')
    parser.add_argument('--model', default='artifacts/models/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', default='tabular',
                       choices=['tabular', 'vision', 'yolo'],
                       help='Type of model')
    parser.add_argument('--config', default='configs/model_config.yaml',
                       help='Model configuration file')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create app
    app = create_app(args.model, args.model_type, config)
    
    # Run server
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Model: {args.model} (type: {args.model_type})")
    print(f"Docs available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == '__main__':
    main()
