#!/usr/bin/env python3
"""
generate_readme_plots.py

Generates publication-quality plots from experiment results for README.md.
Outputs PNG images to paper/figures/.

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


def plot_pretrain_loss():
    """Bar chart: MLM validation loss across 3 runs."""
    runs = ['Run #1\nBERT-Mini\nseq=64', 'Run #2\nBERT-base\nseq=128', 'Run #3\nBERT-base\nseq=512']
    vanilla_loss   = [6.61866, 6.61071, 6.60792]
    tokendrop_loss = [6.61495, 6.61564, 6.60711]
    progdrop_loss  = [6.61557, 6.60477, 6.60241]

    x = np.arange(len(runs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, vanilla_loss,   width, label='Vanilla',   color=C_VANILLA,   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,         tokendrop_loss,  width, label='TokenDrop', color=C_TOKENDROP, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, progdrop_loss,   width, label='ProgDrop',  color=C_PROGDROP,  edgecolor='white', linewidth=0.5)

    ax.set_ylabel('MLM Validation Loss', fontsize=12)
    ax.set_title('Pre-training Loss Across Scales', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=10)
    ax.set_ylim(6.600, 6.622)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Value labels on bars
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


def plot_throughput():
    """Bar chart: throughput comparison at seq=512."""
    steps_per_sec = [7.326, 3.742, 6.241]
    ratios = [1.00, 0.511, 0.852]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(MODELS, steps_per_sec, color=COLORS, edgecolor='white', linewidth=0.5, width=0.55)

    ax.set_ylabel('Steps / Second', fontsize=12)
    ax.set_title('Throughput at seq=512 (BERT-base, batch=8)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        label = f'{height:.2f}\n({ratio:.0%})'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'throughput_seq512.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


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


def plot_scale_trend():
    """Line chart: ProgDrop advantage trend across scales."""
    seq_lens = [64, 128, 512]
    prog_delta_loss = [-0.00309, -0.00594, -0.00552]
    td_delta_loss   = [-0.00371, +0.00493, -0.00082]
    prog_speed_ratio = [0.873, 0.803, 0.852]
    td_speed_ratio   = [0.908, 0.769, 0.511]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Loss delta vs Vanilla
    ax1.plot(seq_lens, prog_delta_loss, 'o-', color=C_PROGDROP, linewidth=2, markersize=8, label='ProgDrop')
    ax1.plot(seq_lens, td_delta_loss,   's--', color=C_TOKENDROP, linewidth=2, markersize=8, label='TokenDrop')
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Sequence Length', fontsize=11)
    ax1.set_ylabel('Loss vs Vanilla (negative = better)', fontsize=11)
    ax1.set_title('Quality Gap Across Scales', fontsize=12, fontweight='bold')
    ax1.set_xticks(seq_lens)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Speed ratio vs Vanilla
    ax2.plot(seq_lens, prog_speed_ratio, 'o-', color=C_PROGDROP, linewidth=2, markersize=8, label='ProgDrop')
    ax2.plot(seq_lens, td_speed_ratio,   's--', color=C_TOKENDROP, linewidth=2, markersize=8, label='TokenDrop')
    ax2.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    ax2.set_xlabel('Sequence Length', fontsize=11)
    ax2.set_ylabel('Speed Ratio vs Vanilla', fontsize=11)
    ax2.set_title('Throughput Ratio Across Scales', fontsize=12, fontweight='bold')
    ax2.set_xticks(seq_lens)
    ax2.set_ylim(0.4, 1.0)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'scale_trend.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generating README plots...\n')
    plot_pretrain_loss()
    plot_throughput()
    plot_sst2()
    plot_scale_trend()
    print(f'\nAll plots saved to {os.path.abspath(FIGURES_DIR)}')
