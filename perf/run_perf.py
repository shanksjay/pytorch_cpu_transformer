#!/usr/bin/env python3
"""Profile transformer runs across model presets and sequence lengths.

Outputs a JSON and CSV summary with MFLOPs, total CPU time (s), total memory (MB),
and top-5 CPU-consuming ops per run.

Usage example:
  python perf/run_perf.py --device cpu --batch 1 --use-reduced

Note: this script requires PyTorch with profiler support installed locally.
"""
import argparse
import json
import csv
from pathlib import Path
from typing import List
import sys
import warnings

# Suppress PyTorch profiler and memory allocation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.profiler.profiler')
warnings.filterwarnings('ignore', message='.*Memory block of unknown size.*')

# Ensure the repository root is on sys.path so imports like `transformer_llama` resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.profiler import profile, ProfilerActivity

from transformer_llama import LlamaConfig, LlamaModel


SEQ_LENS = [64, 128, 256, 512, 1024, 2048, 4096]
PRESETS = [
    ("llama-3.2-1b", "1B"),
    ("llama-3.1-8b", "8B"),
]


def safe_getattr(obj, name, default=0):
    return getattr(obj, name, default)


def run_profile_once(preset_name: str, seq_len: int, batch: int, device: torch.device, use_reduced: bool):
    cfg = LlamaConfig.from_pretrained(preset_name, use_reduced=use_reduced)
    cfg.use_sdpa = False  # keep CPU deterministic profiling

    model = LlamaModel(cfg).to(device)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    attn_mask = None

    # Warmup small number to stabilize timings
    with torch.no_grad():
        for _ in range(1):
            _ = model(input_ids, attn_mask=attn_mask)

    activities = [ProfilerActivity.CPU]
    if device.type == "mps":
        activities.append(ProfilerActivity.MPS)
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True, with_flops=True, acc_events=True) as prof:
        with torch.no_grad():
            logits, _ = model(input_ids, attn_mask=attn_mask)

    ka = prof.key_averages()

    # total flops (may be 0 if not available) in FLOPs
    total_flops = 0
    total_cpu_time_us = 0.0
    total_cpu_mem_bytes = 0
    for op in ka:
        total_flops += safe_getattr(op, "flops", 0)
        total_cpu_time_us += safe_getattr(op, "self_cpu_time_total", 0)
        # try a few attribute names for memory (different torch versions)
        total_cpu_mem_bytes += safe_getattr(op, "self_cpu_memory_usage", 0) or safe_getattr(op, "cpu_memory_usage", 0) or 0

    # convert units
    mflops = float(total_flops) / 1e6
    cpu_time_s = float(total_cpu_time_us) / 1e6  # profiler reports microseconds
    mem_mb = float(total_cpu_mem_bytes) / (1024.0 * 1024.0)

    # top 5 ops by CPU time
    ops = sorted(ka, key=lambda x: safe_getattr(x, "self_cpu_time_total", 0), reverse=True)[:5]
    top_ops = []
    for op in ops:
        name = op.key
        op_time_s = safe_getattr(op, "self_cpu_time_total", 0) / 1e6
        op_pct = (safe_getattr(op, "self_cpu_time_total", 0) / total_cpu_time_us * 100.0) if total_cpu_time_us > 0 else 0.0
        op_flops = safe_getattr(op, "flops", 0)
        op_mem = safe_getattr(op, "self_cpu_memory_usage", 0) or safe_getattr(op, "cpu_memory_usage", 0) or 0
        top_ops.append({
            "op": name,
            "cpu_time_s": float(op_time_s),
            "cpu_pct": float(op_pct),
            "flops": int(op_flops),
            "mem_mb": float(op_mem) / (1024.0 * 1024.0),
        })

    result = {
        "preset": preset_name,
        "model_label": getattr(cfg, "label", preset_name),
        "model_size": next((tag for (n, tag) in PRESETS if n == preset_name), "unknown"),
        "seq_len": seq_len,
        "batch": batch,
        "mflops": float(mflops),
        "total_cpu_time_s": float(cpu_time_s),
        "total_mem_mb": float(mem_mb),
        "top_ops": top_ops,
    }
    return result


def run_profile_avg(preset_name: str, seq_len: int, batch: int, device: torch.device, use_reduced: bool, iters: int = 3):
    runs = []
    for i in range(iters):
        print(f"  Iter {i+1}/{iters}")
        r = run_profile_once(preset_name, seq_len, batch, device, use_reduced)
        runs.append(r)

    # average scalars
    avg_mflops = sum(r["mflops"] for r in runs) / len(runs)
    avg_cpu = sum(r["total_cpu_time_s"] for r in runs) / len(runs)
    avg_mem = sum(r["total_mem_mb"] for r in runs) / len(runs)

    # aggregate ops by name across runs
    op_acc = {}
    for r in runs:
        for op in r.get("top_ops", []):
            name = op.get("op", "")
            if name == "":
                continue
            entry = op_acc.setdefault(name, {"cpu_time_s": 0.0, "flops": 0, "mem_mb": 0.0, "count": 0})
            entry["cpu_time_s"] += op.get("cpu_time_s", 0.0)
            entry["flops"] += op.get("flops", 0)
            entry["mem_mb"] += op.get("mem_mb", 0.0)
            entry["count"] += 1

    averaged_ops = []
    for name, vals in op_acc.items():
        avg_time = vals["cpu_time_s"] / vals["count"]
        avg_flops = int(vals["flops"] / vals["count"])
        avg_mem_mb = vals["mem_mb"] / vals["count"]
        pct = (avg_time / avg_cpu * 100.0) if avg_cpu > 0 else 0.0
        averaged_ops.append({"op": name, "cpu_time_s": avg_time, "cpu_pct": pct, "flops": avg_flops, "mem_mb": avg_mem_mb})

    averaged_ops.sort(key=lambda x: x["cpu_time_s"], reverse=True)
    top_ops = averaged_ops[:5]

    result = {
        "preset": runs[0]["preset"],
        "model_label": runs[0].get("model_label"),
        "model_size": runs[0].get("model_size"),
        "seq_len": seq_len,
        "batch": batch,
        "mflops": float(avg_mflops),
        "total_cpu_time_s": float(avg_cpu),
        "total_mem_mb": float(avg_mem),
        "top_ops": top_ops,
        "raw_runs": runs,
    }
    return result


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3, help="Number of iterations to average per configuration")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Use full-size presets instead of reduced testing presets (default: use reduced)")
    parser.add_argument("--out-dir", type=str, default="perf")
    parser.add_argument("--seq-lens", type=int, nargs="*", default=SEQ_LENS,
                        help="Sequence lengths to benchmark")
    parser.add_argument("--presets", type=str, nargs="*", default=[p[0] for p in PRESETS])
    parser.add_argument("--max-seq-len-override", type=int, default=None,
                        help="Override model max_seq_len (useful for benchmarking longer sequences)")
    args = parser.parse_args(argv)

    # device selection
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for preset in args.presets:
        for seq in args.seq_lens:
            # Load config first to check max_seq_len
            cfg = LlamaConfig.from_pretrained(preset, use_reduced=not args.full, 
                                              max_seq_len=args.max_seq_len_override)
            
            # Skip if seq_len exceeds model's max_seq_len
            if seq > cfg.max_seq_len:
                print(f"Skipping preset={preset} seq_len={seq} (exceeds max_seq_len={cfg.max_seq_len})")
                continue
            
            print(f"Running preset={preset} seq_len={seq} batch={args.batch} device={device}")
            try:
                r = run_profile_avg(preset, seq, args.batch, device, use_reduced=not args.full, iters=args.iters)
                results.append(r)
                # save intermediate results to be safe
                with open(out_dir / "perf_results.json", "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"Run failed for {preset} seq={seq}: {e}")

    # write CSV summary (one row per run, flatten top ops into columns)
    csv_path = out_dir / "perf_summary.csv"
    with open(csv_path, "w", newline="") as cf:
        fieldnames = ["preset", "model_label", "model_size", "seq_len", "batch", "mflops", "total_cpu_time_s", "total_mem_mb"]
        # add top ops columns
        for i in range(1, 6):
            fieldnames += [f"top{i}_op", f"top{i}_cpu_s", f"top{i}_cpu_pct", f"top{i}_flops", f"top{i}_mem_mb"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in ["preset", "model_label", "model_size", "seq_len", "batch", "mflops", "total_cpu_time_s", "total_mem_mb"]}
            for i in range(5):
                top = r["top_ops"][i] if i < len(r["top_ops"]) else {"op": "", "cpu_time_s": 0.0, "cpu_pct": 0.0, "flops": 0, "mem_mb": 0.0}
                row[f"top{i+1}_op"] = top["op"]
                row[f"top{i+1}_cpu_s"] = top["cpu_time_s"]
                row[f"top{i+1}_cpu_pct"] = top["cpu_pct"]
                row[f"top{i+1}_flops"] = top["flops"]
                row[f"top{i+1}_mem_mb"] = top["mem_mb"]
            writer.writerow(row)

    print(f"Wrote results to {out_dir / 'perf_results.json'} and {csv_path}")


if __name__ == "__main__":
    main()
