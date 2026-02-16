#!/usr/bin/env python3
"""
Report Generation Script

Generates training and evaluation reports
"""

import argparse
import sys
from pathlib import Path
import json


def main():
    """Main report generation function"""
    parser = argparse.ArgumentParser(description='Generate training report')
    parser.add_argument(
        '--results-dir',
        default='artifacts/evaluation',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--model-dir',
        default='artifacts/models',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--output',
        help='Output file for report (default: stdout)'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'html', 'json'],
        default='markdown',
        help='Report format'
    )
    
    args = parser.parse_args()
    
    # Load results
    results_file = Path(args.results_dir) / 'evaluation_results.json'
    
    metrics = {}
    if results_file.exists():
        with open(results_file, 'r') as f:
            metrics = json.load(f)
    
    # Generate report
    if args.format == 'markdown':
        report = generate_markdown_report(metrics, args.model_dir)
    elif args.format == 'html':
        report = generate_html_report(metrics, args.model_dir)
    elif args.format == 'json':
        report = json.dumps(metrics, indent=2)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


def generate_markdown_report(metrics: dict, model_dir: str) -> str:
    """Generate markdown report"""
    report = []
    
    report.append("# Model Training Report")
    report.append("")
    report.append("## Model Information")
    report.append(f"- Model Directory: `{model_dir}`")
    report.append("")
    
    report.append("## Evaluation Metrics")
    report.append("")
    
    if metrics:
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                report.append(f"| {key} | {value:.4f} |")
            else:
                report.append(f"| {key} | {value} |")
    else:
        report.append("No metrics available")
    
    report.append("")
    report.append("## Summary")
    report.append("")
    
    if 'mAP50-95' in metrics:
        mAP = metrics['mAP50-95']
        report.append(f"- Overall mAP@0.5:0.95: **{mAP:.4f}**")
        
        if mAP >= 0.5:
            report.append("- Status: ✅ **Excellent performance**")
        elif mAP >= 0.3:
            report.append("- Status: ⚠️ **Good performance**")
        else:
            report.append("- Status: ❌ **Needs improvement**")
    
    return "\n".join(report)


def generate_html_report(metrics: dict, model_dir: str) -> str:
    """Generate HTML report"""
    html = ["<!DOCTYPE html>"]
    html.append("<html>")
    html.append("<head><title>Model Training Report</title></head>")
    html.append("<body>")
    html.append("<h1>Model Training Report</h1>")
    
    html.append("<h2>Model Information</h2>")
    html.append(f"<p>Model Directory: <code>{model_dir}</code></p>")
    
    html.append("<h2>Evaluation Metrics</h2>")
    
    if metrics:
        html.append("<table border='1'>")
        html.append("<tr><th>Metric</th><th>Value</th></tr>")
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                html.append(f"<tr><td>{key}</td><td>{value:.4f}</td></tr>")
            else:
                html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
        
        html.append("</table>")
    else:
        html.append("<p>No metrics available</p>")
    
    html.append("</body>")
    html.append("</html>")
    
    return "\n".join(html)


if __name__ == '__main__':
    main()
