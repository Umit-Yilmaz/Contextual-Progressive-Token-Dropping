#!/usr/bin/env python3
"""
generate_report_plots.py

Generates plots specifically for report-v5.MD (weekly advisor report).
Reads actual training CSVs for epoch-by-epoch curves.
Outputs PNG images to paper/figures/report/.

Usage:
    python analysis/generate_report_plots.py
"""

import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE   = os.path.join(os.path.dirname(__file__), '..')
FIGDIR = os.path.join(BASE, 'paper', 'figures', 'report')
RESULTS = os.path.join(BASE, 'results')
os.makedirs(FIGDIR, exist_ok=True)

C_VANILLA   = '#5B9BD5'
C_TOKENDROP = '#ED7D31'
C_PROGDROP  = '#70AD47'
COLORS = [C_VANILLA, C_TOKENDROP, C_PROGDROP]
MODELS = ['Vanilla', 'TokenDrop', 'ProgDrop']
MODEL_KEYS = ['vanilla', 'tokendrop', 'progressive']


def read_training_csv(run_dir, model_key):
    """Read training-{model}.csv and return list of dicts."""
    path = os.path.join(run_dir, f'training-{model_key}.csv')
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ============================================================
#  Plot 1: Epoch-by-epoch validation loss curves (all 3 runs)
# ============================================================

def plot_epoch_curves():
    """Line plots: val_loss per epoch for each run, all 3 models."""
    runs = [
        ('run1_pilot', 'Run #1 — BERT-Mini, seq=64'),
        ('run2_short', 'Run #2 — BERT-base, seq=128'),
        ('run3_scale', 'Run #3 — BERT-base, seq=512'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (run_name, title) in enumerate(runs):
        ax = axes[idx]
        run_dir = os.path.join(RESULTS, run_name)

        for mk, ml, color in zip(MODEL_KEYS, MODELS, COLORS):
            rows = read_training_csv(run_dir, mk)
            if not rows:
                continue
            epochs = []
            val_losses = []
            for r in rows:
                try:
                    ep = int(r.get('epoch', 0))
                    vl = float(r.get('val_loss', 0))
                    if vl > 0:
                        epochs.append(ep)
                        val_losses.append(vl)
                except (ValueError, TypeError):
                    continue
            if epochs:
                ax.plot(epochs, val_losses, 'o-', color=color, linewidth=1.8,
                        markersize=4, label=ml)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Validation Loss', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Epoch-by-Epoch Validation Loss', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGDIR, 'epoch_val_loss_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 2: Epoch-by-epoch train loss curves (all 3 runs)
# ============================================================

def plot_epoch_train_curves():
    """Line plots: train_loss per epoch for each run, all 3 models."""
    runs = [
        ('run1_pilot', 'Run #1 — BERT-Mini, seq=64'),
        ('run2_short', 'Run #2 — BERT-base, seq=128'),
        ('run3_scale', 'Run #3 — BERT-base, seq=512'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (run_name, title) in enumerate(runs):
        ax = axes[idx]
        run_dir = os.path.join(RESULTS, run_name)

        for mk, ml, color in zip(MODEL_KEYS, MODELS, COLORS):
            rows = read_training_csv(run_dir, mk)
            if not rows:
                continue
            epochs = []
            train_losses = []
            for r in rows:
                try:
                    ep = int(r.get('epoch', 0))
                    tl = float(r.get('train_loss', 0))
                    if tl > 0:
                        epochs.append(ep)
                        train_losses.append(tl)
                except (ValueError, TypeError):
                    continue
            if epochs:
                ax.plot(epochs, train_losses, 'o-', color=color, linewidth=1.8,
                        markersize=4, label=ml)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Train Loss', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Epoch-by-Epoch Training Loss', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGDIR, 'epoch_train_loss_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 3: Combined val + test loss bar chart (Run #3 focus)
# ============================================================

def plot_val_test_comparison():
    """Grouped bar chart: val vs test loss for Run #3 (the main result)."""
    val_losses  = [6.612780, 6.605933, 6.600502]
    test_losses = [6.615509, 6.611065, 6.603808]

    x = np.arange(len(MODELS))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, val_losses,  width, label='Validation', color=COLORS, edgecolor='white', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, test_losses, width, label='Test', color=COLORS, edgecolor='black', linewidth=0.8, alpha=0.55, hatch='//')

    ax.set_ylabel('MLM Loss', fontsize=12)
    ax.set_title('Run #3 — Validation vs Test Loss (BERT-base, seq=512)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylim(6.598, 6.620)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'run3_val_test.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 4: FLOPS comparison (3 configs side by side)
# ============================================================

def plot_flops_report():
    """Bar chart: FLOPS savings for all 3 configs."""
    configs = ['Run #1\nMini, seq=64', 'Run #2\nBase, seq=128', 'Run #3\nBase, seq=512']

    td_savings = [23.3, 24.6, 25.4]
    pd_savings = [19.2, 31.6, 32.7]

    x = np.arange(len(configs))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, td_savings, width, label='TokenDrop', color=C_TOKENDROP, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, pd_savings, width, label='ProgDrop',  color=C_PROGDROP,  edgecolor='white', linewidth=0.5)

    ax.set_ylabel('FLOP Savings vs Vanilla (%)', fontsize=12)
    ax.set_title('Theoretical FLOP Savings by Configuration', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0, 42)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'flops_savings.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 5: Latency comparison
# ============================================================

def plot_latency_report():
    """Bar chart: forward latency across 3 runs."""
    runs_labels = ['Run #1\nMini, seq=64', 'Run #2\nBase, seq=128', 'Run #3\nBase, seq=512']
    vanilla_lat   = [68.164, 140.909, 160.754]
    tokendrop_lat = [75.312, 152.176, 154.127]
    progdrop_lat  = [97.527, 191.001, 171.397]

    x = np.arange(len(runs_labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, vanilla_lat,   width, label='Vanilla',   color=C_VANILLA,   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,         tokendrop_lat,  width, label='TokenDrop', color=C_TOKENDROP, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, progdrop_lat,   width, label='ProgDrop',  color=C_PROGDROP,  edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Forward Latency (ms)', fontsize=12)
    ax.set_title('Forward Latency Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(runs_labels, fontsize=10)
    ax.legend(fontsize=10)
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
    path = os.path.join(FIGDIR, 'latency_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 6: Best epoch convergence
# ============================================================

def plot_convergence():
    """Horizontal bar chart showing best epoch for each model per run."""
    runs = ['Run #1 (seq=64)', 'Run #2 (seq=128)', 'Run #3 (seq=512)']
    vanilla_ep   = [22, 8, 3]
    tokendrop_ep = [25, 4, 3]
    progdrop_ep  = [20, 4, 3]

    y = np.arange(len(runs))
    height = 0.22

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(y - height, vanilla_ep,   height, label='Vanilla',   color=C_VANILLA,   edgecolor='white')
    ax.barh(y,          tokendrop_ep,  height, label='TokenDrop', color=C_TOKENDROP, edgecolor='white')
    ax.barh(y + height, progdrop_ep,   height, label='ProgDrop',  color=C_PROGDROP,  edgecolor='white')

    ax.set_xlabel('Best Epoch (lower = faster convergence)', fontsize=11)
    ax.set_title('Convergence Speed — Best Validation Loss Epoch', fontsize=12, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(runs, fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'convergence_best_epoch.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
#  Plot 7: Summary — ProgDrop advantage across scales
# ============================================================

def plot_summary_advantage():
    """Combined metric: quality + efficiency summary for ProgDrop."""
    seq_lens = [64, 128, 512]

    # Val loss improvement vs Vanilla (negative = better for ProgDrop)
    pd_val_delta = [6.622942 - 6.622923, 6.605759 - 6.601494, 6.600502 - 6.612780]
    pd_flops_saving = [19.2, 31.6, 32.7]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: val loss delta
    bar_colors = [C_PROGDROP if d <= 0 else '#FF6B6B' for d in pd_val_delta]
    bars = ax1.bar([str(s) for s in seq_lens], pd_val_delta, color=bar_colors,
                   edgecolor='white', width=0.5)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Sequence Length', fontsize=11)
    ax1.set_ylabel('Val Loss Difference (ProgDrop - Vanilla)', fontsize=11)
    ax1.set_title('ProgDrop Quality vs Vanilla\n(negative = ProgDrop better)', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -12
        ax1.annotate(f'{height:+.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=10, fontweight='bold')

    # Right: FLOPS saving
    bars2 = ax2.bar([str(s) for s in seq_lens], pd_flops_saving, color=C_PROGDROP,
                    edgecolor='white', width=0.5)
    ax2.set_xlabel('Sequence Length', fontsize=11)
    ax2.set_ylabel('FLOP Savings vs Vanilla (%)', fontsize=11)
    ax2.set_title('ProgDrop Efficiency Gain\n(higher = more savings)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 42)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'progdrop_advantage_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generating report-v5 plots...\n')
    plot_epoch_curves()
    plot_epoch_train_curves()
    plot_val_test_comparison()
    plot_flops_report()
    plot_latency_report()
    plot_convergence()
    plot_summary_advantage()
    print(f'\nAll report plots saved to {os.path.abspath(FIGDIR)}')
