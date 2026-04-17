#!/usr/bin/env python3
"""
generate_readme_plots.py

Generates publication-quality plots from experiment results for README.md.
Outputs PNG images to paper/figures/.

All data below comes from actual training runs on RTX A6000 (48 GB).
FLOPS values are theoretical, computed via scripts/compute_flops.py.

Usage:
    python analysis/generate_readme_plots.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color palette
C_VANILLA   = '#5B9BD5'
C_TOKENDROP = '#ED7D31'
C_PROGDROP  = '#70AD47'

MODELS = ['Vanilla', 'TokenDrop', 'ProgDrop']
COLORS = [C_VANILLA, C_TOKENDROP, C_PROGDROP]

# ============================================================
#  Actual experiment results (from results_summary.json)
# ============================================================

# Run #1 — Pilot: BERT-Mini (L=4, d=256, h=4, ff=1024), seq=64, 25 epochs, batch=256
RUN1 = {
    'label': 'Run #1\nBERT-Mini\nseq=64',
    'vanilla':   {'val': 6.622923, 'test': 6.617263, 'train': 6.470076, 'best_epoch': 22, 'steps_s': 6.693, 'latency_ms': 68.164},
    'tokendrop': {'val': 6.624680, 'test': 6.618723, 'train': 6.469548, 'best_epoch': 25, 'steps_s': 6.087, 'latency_ms': 75.312},
    'progdrop':  {'val': 6.622942, 'test': 6.617642, 'train': 6.468651, 'best_epoch': 20, 'steps_s': 5.816, 'latency_ms': 97.527},
}

# Run #2 — Short: BERT-base (L=12, d=768, h=12, ff=3072), seq=128, 50K steps, batch=64
RUN2 = {
    'label': 'Run #2\nBERT-base\nseq=128',
    'vanilla':   {'val': 6.601494, 'test': 6.603011, 'train': 6.402123, 'best_epoch': 8,  'steps_s': 4.549, 'latency_ms': 140.909},
    'tokendrop': {'val': 6.613844, 'test': 6.615121, 'train': 6.377631, 'best_epoch': 4,  'steps_s': 3.491, 'latency_ms': 152.176},
    'progdrop':  {'val': 6.605759, 'test': 6.606361, 'train': 6.365768, 'best_epoch': 4,  'steps_s': 3.645, 'latency_ms': 191.001},
}

# Run #3 — Scale: BERT-base (L=12, d=768, h=12, ff=3072), seq=512, 200K steps, batch=16
RUN3 = {
    'label': 'Run #3\nBERT-base\nseq=512',
    'vanilla':   {'val': 6.612780, 'test': 6.615509, 'train': 6.389848, 'best_epoch': 3,  'steps_s': 4.043, 'latency_ms': 160.754},
    'tokendrop': {'val': 6.605933, 'test': 6.611065, 'train': 6.356227, 'best_epoch': 3,  'steps_s': 3.223, 'latency_ms': 154.127},
    'progdrop':  {'val': 6.600502, 'test': 6.603808, 'train': 6.354696, 'best_epoch': 3,  'steps_s': 3.426, 'latency_ms': 171.397},
}

RUNS = [RUN1, RUN2, RUN3]

# ============================================================
#  Theoretical FLOPS (from scripts/compute_flops.py)
# ============================================================

FLOPS = {
    'run1': {'vanilla': 419.43e6, 'tokendrop': 321.91e6, 'progdrop': 338.95e6,
             'label': 'BERT-Mini, seq=64'},
    'run2': {'vanilla': 22.35e9, 'tokendrop': 16.85e9, 'progdrop': 15.28e9,
             'label': 'BERT-base, seq=128'},
    'run3': {'vanilla': 96.64e9, 'tokendrop': 72.07e9, 'progdrop': 65.08e9,
             'label': 'BERT-base, seq=512'},
}


# ============================================================
#  Plot 1: Pre-training Validation Loss (3 runs)
# ============================================================

def plot_pretrain_loss():
    """Bar chart: MLM validation loss across 3 runs."""
    runs_labels = [r['label'] for r in RUNS]
    vanilla_loss   = [r['vanilla']['val'] for r in RUNS]
    tokendrop_loss = [r['tokendrop']['val'] for r in RUNS]
    progdrop_loss  = [r['progdrop']['val'] for r in RUNS]

    x = np.arange(len(runs_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, vanilla_loss,   width, label='Vanilla',   color=C_VANILLA,   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,         tokendrop_loss,  width, label='TokenDrop', color=C_TOKENDROP, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, progdrop_loss,   width, label='ProgDrop',  color=C_PROGDROP,  edgecolor='white', linewidth=0.5)

    ax.set_ylabel('MLM Validation Loss', fontsize=12)
    ax.set_title('Pre-training Validation Loss Across Scales', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(runs_labels, fontsize=10)
    ax.set_ylim(6.598, 6.628)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'pretrain_loss.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 2: Test Loss (3 runs)
# ============================================================

def plot_test_loss():
    """Bar chart: MLM test loss across 3 runs."""
    runs_labels = [r['label'] for r in RUNS]
    vanilla_loss   = [r['vanilla']['test'] for r in RUNS]
    tokendrop_loss = [r['tokendrop']['test'] for r in RUNS]
    progdrop_loss  = [r['progdrop']['test'] for r in RUNS]

    x = np.arange(len(runs_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, vanilla_loss,   width, label='Vanilla',   color=C_VANILLA,   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,         tokendrop_loss,  width, label='TokenDrop', color=C_TOKENDROP, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, progdrop_loss,   width, label='ProgDrop',  color=C_PROGDROP,  edgecolor='white', linewidth=0.5)

    ax.set_ylabel('MLM Test Loss', fontsize=12)
    ax.set_title('Test Loss Across Scales', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(runs_labels, fontsize=10)
    ax.set_ylim(6.598, 6.622)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'test_loss.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 3: Forward Latency comparison (3 runs)
# ============================================================

def plot_latency():
    """Grouped bar chart: forward latency (ms) across 3 runs."""
    runs_labels = [r['label'] for r in RUNS]
    vanilla_lat   = [r['vanilla']['latency_ms'] for r in RUNS]
    tokendrop_lat = [r['tokendrop']['latency_ms'] for r in RUNS]
    progdrop_lat  = [r['progdrop']['latency_ms'] for r in RUNS]

    x = np.arange(len(runs_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, vanilla_lat,   width, label='Vanilla',   color=C_VANILLA,   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,         tokendrop_lat,  width, label='TokenDrop', color=C_TOKENDROP, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, progdrop_lat,   width, label='ProgDrop',  color=C_PROGDROP,  edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Forward Latency (ms)', fontsize=12)
    ax.set_title('Forward Latency Across Scales', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(runs_labels, fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'latency_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 4: Theoretical FLOPS comparison (3 configs)
# ============================================================

def plot_flops():
    """Grouped bar chart: theoretical FLOPs across 3 configs."""
    configs = ['Run #1\nMini, seq=64', 'Run #2\nBase, seq=128', 'Run #3\nBase, seq=512']
    keys = ['run1', 'run2', 'run3']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (key, cfg_label) in enumerate(zip(keys, configs)):
        ax = axes[idx]
        vals = [FLOPS[key]['vanilla'], FLOPS[key]['tokendrop'], FLOPS[key]['progdrop']]
        ratios = [v / vals[0] for v in vals]

        # Choose unit
        if vals[0] >= 1e9:
            display = [v / 1e9 for v in vals]
            unit = 'GFLOPS'
        else:
            display = [v / 1e6 for v in vals]
            unit = 'MFLOPS'

        bars = ax.bar(MODELS, display, color=COLORS, edgecolor='white',
                      linewidth=0.5, width=0.55)
        ax.set_ylabel(unit, fontsize=11)
        ax.set_title(cfg_label, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            label = f'{height:.1f}\n({ratio:.0%})'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Theoretical FLOPs Comparison (per forward pass)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'flops_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 5: Best Epoch comparison
# ============================================================

def plot_best_epoch():
    """Bar chart: best epoch (early stop point) across 3 runs."""
    runs_labels = [r['label'] for r in RUNS]
    vanilla_ep   = [r['vanilla']['best_epoch'] for r in RUNS]
    tokendrop_ep = [r['tokendrop']['best_epoch'] for r in RUNS]
    progdrop_ep  = [r['progdrop']['best_epoch'] for r in RUNS]

    x = np.arange(len(runs_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, vanilla_ep,   width, label='Vanilla',   color=C_VANILLA,   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,         tokendrop_ep,  width, label='TokenDrop', color=C_TOKENDROP, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, progdrop_ep,   width, label='ProgDrop',  color=C_PROGDROP,  edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Best Epoch', fontsize=12)
    ax.set_title('Convergence Speed (Best Epoch)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(runs_labels, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'best_epoch.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 6: Scale Trend (quality gap + speed ratio)
# ============================================================

def plot_scale_trend():
    """Line chart: ProgDrop advantage trend across scales."""
    seq_lens = [64, 128, 512]

    prog_delta_val  = [r['progdrop']['val'] - r['vanilla']['val'] for r in RUNS]
    td_delta_val    = [r['tokendrop']['val'] - r['vanilla']['val'] for r in RUNS]
    prog_speed_ratio = [r['progdrop']['steps_s'] / r['vanilla']['steps_s'] for r in RUNS]
    td_speed_ratio   = [r['tokendrop']['steps_s'] / r['vanilla']['steps_s'] for r in RUNS]

    # FLOPS savings
    flops_keys = ['run1', 'run2', 'run3']
    prog_flops_ratio = [FLOPS[k]['progdrop'] / FLOPS[k]['vanilla'] for k in flops_keys]
    td_flops_ratio   = [FLOPS[k]['tokendrop'] / FLOPS[k]['vanilla'] for k in flops_keys]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Loss delta vs Vanilla
    ax1.plot(seq_lens, prog_delta_val, 'o-', color=C_PROGDROP, linewidth=2, markersize=8, label='ProgDrop')
    ax1.plot(seq_lens, td_delta_val,   's--', color=C_TOKENDROP, linewidth=2, markersize=8, label='TokenDrop')
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Sequence Length', fontsize=11)
    ax1.set_ylabel('Val Loss vs Vanilla (negative = better)', fontsize=11)
    ax1.set_title('Quality Gap Across Scales', fontsize=12, fontweight='bold')
    ax1.set_xticks(seq_lens)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Middle: Speed ratio vs Vanilla
    ax2.plot(seq_lens, prog_speed_ratio, 'o-', color=C_PROGDROP, linewidth=2, markersize=8, label='ProgDrop')
    ax2.plot(seq_lens, td_speed_ratio,   's--', color=C_TOKENDROP, linewidth=2, markersize=8, label='TokenDrop')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Sequence Length', fontsize=11)
    ax2.set_ylabel('Speed Ratio vs Vanilla', fontsize=11)
    ax2.set_title('Throughput Ratio', fontsize=12, fontweight='bold')
    ax2.set_xticks(seq_lens)
    ax2.set_ylim(0.6, 1.0)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Right: FLOPS ratio vs Vanilla
    ax3.plot(seq_lens, prog_flops_ratio, 'o-', color=C_PROGDROP, linewidth=2, markersize=8, label='ProgDrop')
    ax3.plot(seq_lens, td_flops_ratio,   's--', color=C_TOKENDROP, linewidth=2, markersize=8, label='TokenDrop')
    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Sequence Length', fontsize=11)
    ax3.set_ylabel('FLOPS Ratio vs Vanilla', fontsize=11)
    ax3.set_title('Theoretical FLOPS Ratio', fontsize=12, fontweight='bold')
    ax3.set_xticks(seq_lens)
    ax3.set_ylim(0.5, 1.0)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'scale_trend.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 7: Throughput (steps/s) across all runs
# ============================================================

def plot_throughput():
    """Bar chart: throughput (steps/s) across 3 runs."""
    runs_labels = [r['label'] for r in RUNS]
    vanilla_sps   = [r['vanilla']['steps_s'] for r in RUNS]
    tokendrop_sps = [r['tokendrop']['steps_s'] for r in RUNS]
    progdrop_sps  = [r['progdrop']['steps_s'] for r in RUNS]

    x = np.arange(len(runs_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, vanilla_sps,   width, label='Vanilla',   color=C_VANILLA,   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,         tokendrop_sps,  width, label='TokenDrop', color=C_TOKENDROP, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, progdrop_sps,   width, label='ProgDrop',  color=C_PROGDROP,  edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Steps / Second', fontsize=12)
    ax.set_title('Training Throughput Across Scales', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(runs_labels, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'throughput_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 8 (legacy): SST-2 accuracy
# ============================================================

def plot_sst2():
    """Bar chart: SST-2 downstream accuracy."""
    accuracies = [50.12, 59.84, 62.50]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(MODELS, accuracies, color=COLORS, edgecolor='white', linewidth=0.5, width=0.55)

    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('SST-2 Fine-tune (from 200K-step checkpoints)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 75)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'sst2_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generating README plots...\n')
    plot_pretrain_loss()
    plot_test_loss()
    plot_latency()
    plot_flops()
    plot_best_epoch()
    plot_scale_trend()
    plot_throughput()
    plot_sst2()
    print(f'\nAll plots saved to {os.path.abspath(FIGURES_DIR)}')
