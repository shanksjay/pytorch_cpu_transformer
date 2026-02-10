#!/usr/bin/env python3
"""Generate performance visualization plots from perf_summary.csv.

Produces matplotlib plots showing:
  - MFLOPs scaling across sequence lengths and model sizes
  - CPU time scaling (latency)
  - Memory scaling
  - Top operation breakdown per configuration

Usage:
  python perf/visualize_perf.py
  
Output:
  perf/plots/scaling_mflops.png
  perf/plots/scaling_cpu_time.png
  perf/plots/scaling_memory.png
  perf/plots/top_ops_breakdown.png
"""
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Try importing seaborn for better styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_csv(csv_path: Path) -> List[Dict]:
    """Load perf_summary.csv and return list of dicts."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['seq_len'] = int(row['seq_len'])
            row['num_threads'] = int(row.get('num_threads', 1))
            row['mflops'] = float(row['mflops'])
            row['total_cpu_time_s'] = float(row['total_cpu_time_s'])
            row['total_mem_mb'] = float(row['total_mem_mb'])
            results.append(row)
    return results


def group_by_model(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by model_size (1B, 8B, etc) in order: 1B, 8B, 70B."""
    grouped = {}
    for row in results:
        model = row['model_size']
        if model not in grouped:
            grouped[model] = []
        grouped[model].append(row)
    # Sort each group by seq_len
    for model in grouped:
        grouped[model].sort(key=lambda x: x['seq_len'])
    # Return in custom order: 1B, 8B, 70B
    model_order = ['1B', '8B', '70B']
    return {model: grouped[model] for model in model_order if model in grouped}


def plot_mflops_scaling(grouped: Dict[str, List[Dict]], out_dir: Path):
    """Plot MFLOPs achieved vs sequence length."""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (model, data), color in zip(grouped.items(), colors):
        seq_lens = [d['seq_len'] for d in data]
        mflops = [float(d['mflops']) for d in data]
        ax.plot(seq_lens, mflops, marker='o', linewidth=2.5, markersize=9, 
                label=f"Llama {model}", color=color)
    
    ax.set_xlabel("Sequence Length (log scale)", fontsize=12, fontweight='bold')
    ax.set_ylabel("MFLOPs (Million FLOPs)", fontsize=12, fontweight='bold')
    ax.set_title("Compute Throughput Scaling across Sequence Lengths", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', title='Model Size')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    plt.tight_layout()
    plt.savefig(out_dir / 'scaling_mflops.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_dir / 'scaling_mflops.png'}")


def plot_cpu_time_scaling(grouped: Dict[str, List[Dict]], out_dir: Path):
    """Plot CPU time (latency) vs sequence length."""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (model, data), color in zip(grouped.items(), colors):
        seq_lens = [d['seq_len'] for d in data]
        times = [float(d['total_cpu_time_s']) for d in data]
        ax.plot(seq_lens, times, marker='s', linewidth=2.5, markersize=9, 
                label=f"Llama {model}", color=color)
    
    ax.set_xlabel("Sequence Length (log scale)", fontsize=12, fontweight='bold')
    ax.set_ylabel("CPU Time (seconds, log scale)", fontsize=12, fontweight='bold')
    ax.set_title("Latency Scaling across Sequence Lengths", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', title='Model Size')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    plt.tight_layout()
    plt.savefig(out_dir / 'scaling_cpu_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_dir / 'scaling_cpu_time.png'}")


def plot_memory_scaling(grouped: Dict[str, List[Dict]], out_dir: Path):
    """Plot memory usage vs sequence length."""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (model, data), color in zip(grouped.items(), colors):
        seq_lens = [d['seq_len'] for d in data]
        mems = [float(d['total_mem_mb']) for d in data]
        ax.plot(seq_lens, mems, marker='^', linewidth=2.5, markersize=9, 
                label=f"Llama {model}", color=color)
    
    ax.set_xlabel("Sequence Length (log scale)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Peak Memory (MB, log scale)", fontsize=12, fontweight='bold')
    ax.set_title("Memory Usage Scaling across Sequence Lengths", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', title='Model Size')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    plt.tight_layout()
    plt.savefig(out_dir / 'scaling_memory.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_dir / 'scaling_memory.png'}")


def plot_top_ops_breakdown(grouped: Dict[str, List[Dict]], out_dir: Path):
    """Plot top operations breakdown as stacked bar chart."""
    fig, axes = plt.subplots(1, len(grouped), figsize=(5 * len(grouped), 6))
    if len(grouped) == 1:
        axes = [axes]
    
    colors_ops = plt.cm.Set3(np.linspace(0, 1, 5))
    
    for idx, (model, data) in enumerate(grouped.items()):
        ax = axes[idx]
        
        # Aggregate top ops across all seq_lens for this model
        op_times = {}
        for row in data:
            for i in range(1, 6):
                op_name = row.get(f'top{i}_op', '')
                op_pct = float(row.get(f'top{i}_cpu_pct', 0))
                if op_name:
                    if op_name not in op_times:
                        op_times[op_name] = []
                    op_times[op_name].append(op_pct)
        
        # Average percentages
        avg_op_times = {op: np.mean(times) for op, times in op_times.items()}
        sorted_ops = sorted(avg_op_times.items(), key=lambda x: x[1], reverse=True)[:5]
        
        ops = [op for op, _ in sorted_ops]
        pcts = [pct for _, pct in sorted_ops]
        
        bars = ax.barh(ops, pcts, color=colors_ops[:len(ops)])
        ax.set_xlabel("Average CPU Time (%)", fontsize=11, fontweight='bold')
        ax.set_title(f"Llama {model}\nTop Operations", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, pcts)):
            ax.text(pct + 2, i, f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.suptitle("Top 5 CPU-Consuming Operations by Model", 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(out_dir / 'top_ops_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_dir / 'top_ops_breakdown.png'}")


def plot_thread_scaling_latency(results: List[Dict], out_dir: Path):
    """Plot CPU time vs number of threads for fixed sequence lengths."""
    # Check if we have thread data
    if all(r['num_threads'] == 1 for r in results):
        print("⊘ Skipping thread scaling plots (no multi-thread data available)")
        return
    
    # Group by model and seq_len, then plot latency vs threads
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Get unique models and seq_lens
    models = sorted(set(r['model_size'] for r in results), key=lambda x: ('1B', '8B', '70B').index(x) if x in ('1B', '8B', '70B') else 999)
    seq_lens = sorted(set(r['seq_len'] for r in results))
    
    colors_models = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plot_idx = 0
    for model in models:
        if plot_idx >= 4:
            break
        ax = axes[plot_idx]
        
        model_data = [r for r in results if r['model_size'] == model]
        
        # Get unique thread counts for this model
        thread_counts = sorted(set(r['num_threads'] for r in model_data))
        
        if len(thread_counts) < 2:
            plot_idx += 1
            continue
        
        for seq_len in seq_lens[:3]:  # Plot first 3 seq_lens
            seq_data = [r for r in model_data if r['seq_len'] == seq_len]
            if not seq_data:
                continue
            
            seq_data.sort(key=lambda x: x['num_threads'])
            threads = [r['num_threads'] for r in seq_data]
            times = [r['total_cpu_time_s'] for r in seq_data]
            
            ax.plot(threads, times, marker='o', linewidth=2.5, markersize=8, 
                   label=f"Seq len {seq_len}")
        
        ax.set_xlabel("Number of CPU Threads", fontsize=11, fontweight='bold')
        ax.set_ylabel("Latency (seconds)", fontsize=11, fontweight='bold')
        ax.set_title(f"Llama {model} - CPU Time vs Threads", fontsize=12, fontweight='bold')
        ax.set_xticks(thread_counts)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, 4):
        axes[idx].axis('off')
    
    plt.suptitle("Latency Scaling with CPU Thread Count", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'thread_scaling_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_dir / 'thread_scaling_latency.png'}")


def plot_thread_scaling_speedup(results: List[Dict], out_dir: Path):
    """Plot speedup vs number of threads (baseline = 1 thread)."""
    # Check if we have thread data
    if all(r['num_threads'] == 1 for r in results):
        return
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Get unique models and seq_lens
    models = sorted(set(r['model_size'] for r in results), key=lambda x: ('1B', '8B', '70B').index(x) if x in ('1B', '8B', '70B') else 999)
    
    colors_models = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for model, color in zip(models, colors_models):
        model_data = [r for r in results if r['model_size'] == model]
        
        # Get data at a fixed seq_len (use middle one for fairness)
        seq_lens = sorted(set(r['seq_len'] for r in model_data))
        if len(seq_lens) < 2:
            mid_seq = seq_lens[0]
        else:
            mid_seq = seq_lens[len(seq_lens) // 2]
        
        seq_data = [r for r in model_data if r['seq_len'] == mid_seq]
        seq_data.sort(key=lambda x: x['num_threads'])
        
        if not seq_data:
            continue
        
        # Compute speedup relative to 1 thread
        baseline_time = next((r['total_cpu_time_s'] for r in seq_data if r['num_threads'] == 1), None)
        if baseline_time is None or baseline_time == 0:
            continue
        
        threads = [r['num_threads'] for r in seq_data]
        speedups = [baseline_time / r['total_cpu_time_s'] for r in seq_data]
        
        ax.plot(threads, speedups, marker='o', linewidth=2.5, markersize=9,
               label=f"Llama {model} (seq_len={mid_seq})", color=color)
    
    # Add ideal linear scaling line
    thread_range = range(1, max(threads) + 1 if threads else 2)
    ax.plot(thread_range, thread_range, 'k--', linewidth=2, alpha=0.5, label='Ideal Linear Speedup')
    
    ax.set_xlabel("Number of CPU Threads", fontsize=12, fontweight='bold')
    ax.set_ylabel("Speedup (Relative to 1 Thread)", fontsize=12, fontweight='bold')
    ax.set_title("CPU Parallelization Efficiency - Speedup vs Thread Count", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    if threads:
        ax.set_xticks(sorted(set(threads)))
    
    plt.tight_layout()
    plt.savefig(out_dir / 'thread_scaling_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_dir / 'thread_scaling_speedup.png'}")


def print_summary_table(grouped: Dict[str, List[Dict]]):
    """Print a summary table of key metrics."""
    print("\n" + "="*80)
    print("PERFORMANCE SCALING SUMMARY")
    print("="*80)
    
    for model in grouped.keys():
        data = grouped[model]
        print(f"\n{model} Model:")
        print(f"  {'Seq Len':<10} {'MFLOPs':<15} {'CPU Time (s)':<15} {'Memory (MB)':<15}")
        print("  " + "-"*55)
        for row in data:
            print(f"  {row['seq_len']:<10} {float(row['mflops']):<15.0f} "
                  f"{float(row['total_cpu_time_s']):<15.6f} {float(row['total_mem_mb']):<15.1f}")
        
        # Compute scaling factors
        if len(data) > 1:
            first = data[0]
            last = data[-1]
            mflops_scale = float(last['mflops']) / float(first['mflops'])
            time_scale = float(last['total_cpu_time_s']) / float(first['total_cpu_time_s'])
            mem_scale = float(last['total_mem_mb']) / float(first['total_mem_mb'])
            
            print(f"\n  Scaling factors (seq_len {first['seq_len']} → {last['seq_len']}):")
            print(f"    MFLOPs: {mflops_scale:.1f}x")
            print(f"    CPU Time: {time_scale:.1f}x")
            print(f"    Memory: {mem_scale:.1f}x")


def main():
    csv_path = Path(__file__).parent / "perf_summary.csv"
    out_dir = Path(__file__).parent / "plots"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run perf/run_perf.py first.")
        return 1
    
    out_dir.mkdir(exist_ok=True)
    
    print(f"Loading {csv_path}...")
    results = load_csv(csv_path)
    
    if not results:
        print("No data found in CSV.")
        return 1
    
    grouped = group_by_model(results)
    
    print(f"Found {len(results)} configurations across {len(grouped)} models")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_mflops_scaling(grouped, out_dir)
    plot_cpu_time_scaling(grouped, out_dir)
    plot_memory_scaling(grouped, out_dir)
    plot_top_ops_breakdown(grouped, out_dir)
    
    # Generate thread scaling plots if data available
    plot_thread_scaling_latency(results, out_dir)
    plot_thread_scaling_speedup(results, out_dir)
    
    # Print summary
    print_summary_table(grouped)
    
    print("\n" + "="*80)
    print(f"✓ All plots saved to: {out_dir.absolute()}")
    print("="*80)


if __name__ == "__main__":
    exit(main())
