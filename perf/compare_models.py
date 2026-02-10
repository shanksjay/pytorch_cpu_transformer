#!/usr/bin/env python3
"""Generate comprehensive comparison table across all models and sequence lengths.

Usage:
  python perf/compare_models.py
  
Output:
  perf/comparison_table.txt - formatted text table
  perf/comparison_table.csv - spreadsheet-friendly CSV
"""
import csv
from pathlib import Path
from typing import List, Dict
import json


def load_csv(csv_path: Path) -> List[Dict]:
    """Load perf_summary.csv and return list of dicts."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['seq_len'] = int(row['seq_len'])
            row['num_threads'] = int(row.get('num_threads', 1))
            row['mflops'] = float(row['mflops'])
            row['total_cpu_time_s'] = float(row['total_cpu_time_s'])
            row['total_mem_mb'] = float(row['total_mem_mb'])
            results.append(row)
    
    # Custom sort order: 1B, 8B, 70B
    model_order = {'1B': 0, '8B': 1, '70B': 2}
    return sorted(results, key=lambda x: (model_order.get(x['model_size'], 99), x['seq_len'], x['num_threads']))


def generate_text_table(results: List[Dict], out_path: Path):
    """Generate formatted text table."""
    
    # Group by model
    models = {}
    for row in results:
        model = row['model_size']
        if model not in models:
            models[model] = []
        models[model].append(row)
    
    # Custom sort order: 1B, 8B, 70B
    model_order = ['1B', '8B', '70B']
    sorted_models = [m for m in model_order if m in models]
    
    # Create formatted output
    output = []
    output.append("="*120)
    output.append("TRANSFORMER MODEL PERFORMANCE COMPARISON - ALL CONFIGURATIONS")
    output.append("="*120)
    
    # Create a unified table
    output.append("\n")
    output.append(f"{'Model':<12} {'Seq Len':<12} {'MFLOPs':<15} {'CPU Time (s)':<15} {'Memory (MB)':<15} {'MFLOPs/s':<15} {'MFLOPs/MB':<15}")
    output.append("-"*120)
    
    for model in sorted_models:
        model_data = models[model]
        for row in model_data:
            efficiency = row['mflops'] / row['total_cpu_time_s']
            mem_efficiency = row['mflops'] / row['total_mem_mb']
            output.append(
                f"{row['model_size']:<12} {row['seq_len']:<12} {row['mflops']:<15.0f} "
                f"{row['total_cpu_time_s']:<15.6f} {row['total_mem_mb']:<15.1f} "
                f"{efficiency:<15.0f} {mem_efficiency:<15.2f}"
            )
    
    output.append("="*120)
    
    # Add scaling analysis
    output.append("\nSCALING ANALYSIS (64 → 512 sequence length):\n")
    output.append(f"{'Model':<12} {'MFLOPs Scale':<15} {'CPU Time Scale':<15} {'Memory Scale':<15} {'Efficiency':<15}")
    output.append("-"*72)
    
    for model in sorted_models:
        model_data = sorted(models[model], key=lambda x: x['seq_len'])
        if len(model_data) >= 2:
            first = model_data[0]
            last = model_data[-1]
            
            mflops_scale = last['mflops'] / first['mflops']
            time_scale = last['total_cpu_time_s'] / first['total_cpu_time_s']
            mem_scale = last['total_mem_mb'] / first['total_mem_mb']
            efficiency = mflops_scale / (last['seq_len'] / first['seq_len'])
            
            output.append(
                f"{model:<12} {mflops_scale:<15.2f}x {time_scale:<15.2f}x "
                f"{mem_scale:<15.2f}x {efficiency:<15.2f}x"
            )
    
    output.append("="*120)
    
    # Add summary statistics
    output.append("\nSUMMARY STATISTICS:\n")
    output.append(f"{'Model':<12} {'Avg MFLOPs':<15} {'Avg CPU Time':<15} {'Avg Memory':<15} {'Avg Efficiency':<15}")
    output.append("-"*72)
    
    for model in sorted_models:
        model_data = models[model]
        avg_mflops = sum(r['mflops'] for r in model_data) / len(model_data)
        avg_time = sum(r['total_cpu_time_s'] for r in model_data) / len(model_data)
        avg_mem = sum(r['total_mem_mb'] for r in model_data) / len(model_data)
        avg_eff = sum(r['mflops'] / r['total_cpu_time_s'] for r in model_data) / len(model_data)
        
        output.append(
            f"{model:<12} {avg_mflops:<15.0f} {avg_time:<15.6f} "
            f"{avg_mem:<15.1f} {avg_eff:<15.0f}"
        )
    
    output.append("="*120)
    
    text = "\n".join(output)
    with open(out_path, 'w') as f:
        f.write(text)
    
    print(text)
    return text


def generate_csv_table(results: List[Dict], out_path: Path):
    """Generate CSV for spreadsheet analysis."""
    
    with open(out_path, 'w', newline='') as f:
        fieldnames = [
            'model_size', 'seq_len', 'mflops', 'total_cpu_time_s', 'total_mem_mb',
            'mflops_per_sec', 'mflops_per_mb'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in results:
            efficiency = row['mflops'] / row['total_cpu_time_s']
            mem_efficiency = row['mflops'] / row['total_mem_mb']
            
            writer.writerow({
                'model_size': row['model_size'],
                'seq_len': row['seq_len'],
                'mflops': f"{row['mflops']:.0f}",
                'total_cpu_time_s': f"{row['total_cpu_time_s']:.6f}",
                'total_mem_mb': f"{row['total_mem_mb']:.1f}",
                'mflops_per_sec': f"{efficiency:.0f}",
                'mflops_per_mb': f"{mem_efficiency:.2f}"
            })


def generate_markdown_table(results: List[Dict], out_path: Path):
    """Generate markdown table for documentation."""
    
    # Group by model
    models = {}
    for row in results:
        model = row['model_size']
        if model not in models:
            models[model] = []
        models[model].append(row)
    
    # Custom sort order: 1B, 8B, 70B
    model_order = ['1B', '8B', '70B']
    sorted_models = [m for m in model_order if m in models]
    
    output = []
    output.append("# Performance Comparison Table\n")
    output.append("## All Models Across Sequence Lengths\n")
    
    for model in sorted_models:
        output.append(f"### Llama {model}\n")
        output.append("| Seq Len | MFLOPs | CPU Time (s) | Memory (MB) | MFLOPs/s | MFLOPs/MB |")
        output.append("|---------|--------|--------------|-------------|----------|-----------|")
        
        for row in models[model]:
            efficiency = row['mflops'] / row['total_cpu_time_s']
            mem_efficiency = row['mflops'] / row['total_mem_mb']
            output.append(
                f"| {row['seq_len']} | {row['mflops']:.0f} | {row['total_cpu_time_s']:.6f} | "
                f"{row['total_mem_mb']:.1f} | {efficiency:.0f} | {mem_efficiency:.2f} |"
            )
        
        output.append("")
    
    markdown = "\n".join(output)
    with open(out_path, 'w') as f:
        f.write(markdown)


def generate_thread_scaling_table(results: List[Dict], out_path: Path):
    """Generate thread scaling analysis table."""
    
    # Check if we have multi-threaded data
    thread_counts = sorted(set(r['num_threads'] for r in results))
    if len(thread_counts) < 2:
        return  # Skip if no thread scaling data
    
    # Group by model, seq_len
    data_by_model_seq = {}
    for row in results:
        key = (row['model_size'], row['seq_len'])
        if key not in data_by_model_seq:
            data_by_model_seq[key] = []
        data_by_model_seq[key].append(row)
    
    output = []
    output.append("="*100)
    output.append("CPU THREAD SCALING ANALYSIS - SPEEDUP AND EFFICIENCY")
    output.append("="*100)
    output.append("\nNote: Speedup measured relative to 1-thread baseline")
    output.append("       Efficiency = Speedup / Num_Threads (ideal = 1.0)\n")
    
    # Custom model order
    model_order = ['1B', '8B', '70B']
    
    for model in model_order:
        model_configs = {k: v for k, v in data_by_model_seq.items() if k[0] == model}
        if not model_configs:
            continue
        
        output.append(f"\n{model} Model - Thread Scaling Analysis:")
        output.append("-"*100)
        output.append(f"{'Seq Len':<12} {'Threads':<10} {'Latency (s)':<15} {'Speedup':<12} {'Efficiency':<12} {'MFLOPs':<15}")
        output.append("-"*100)
        
        for (mdl, seq_len), thread_data in sorted(model_configs.items()):
            thread_data.sort(key=lambda x: x['num_threads'])
            
            # Get baseline (1 thread)
            baseline = next((r for r in thread_data if r['num_threads'] == 1), None)
            if not baseline:
                continue
            
            for row in thread_data:
                speedup = baseline['total_cpu_time_s'] / row['total_cpu_time_s']
                efficiency = speedup / row['num_threads']
                output.append(
                    f"{seq_len:<12} {row['num_threads']:<10} {row['total_cpu_time_s']:<15.6f} "
                    f"{speedup:<12.2f}x {efficiency:<12.2f} {row['mflops']:<15.0f}"
                )
            
            output.append("")  # Blank line between seq_lens
    
    output.append("="*100)
    
    with open(out_path, 'w') as f:
        f.write('\n'.join(output))


def main():
    csv_path = Path(__file__).parent / "perf_summary.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run perf/run_perf.py first.")
        return 1
    
    print(f"Loading {csv_path}...\n")
    results = load_csv(csv_path)
    
    if not results:
        print("No data found in CSV.")
        return 1
    
    # Generate outputs
    text_path = Path(__file__).parent / "comparison_table.txt"
    csv_path_out = Path(__file__).parent / "comparison_table.csv"
    md_path = Path(__file__).parent / "comparison_table.md"
    thread_scaling_path = Path(__file__).parent / "thread_scaling_analysis.txt"
    
    print("Generating comparison tables...\n")
    
    # Text table
    generate_text_table(results, text_path)
    print(f"\n✓ Saved text table: {text_path}")
    
    # CSV table
    generate_csv_table(results, csv_path_out)
    print(f"✓ Saved CSV table: {csv_path_out}")
    
    # Markdown table
    generate_markdown_table(results, md_path)
    print(f"✓ Saved markdown table: {md_path}")
    
    # Thread scaling table (if available)
    generate_thread_scaling_table(results, thread_scaling_path)
    if thread_scaling_path.exists() and thread_scaling_path.stat().st_size > 0:
        print(f"✓ Saved thread scaling analysis: {thread_scaling_path}")


if __name__ == "__main__":
    exit(main())
