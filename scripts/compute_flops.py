#!/usr/bin/env python3
"""
compute_flops.py

Computes theoretical FLOPs for the three BERT variants:
  - Vanilla BERT (no token dropping)
  - TokenDrop BERT (single-stage L2-norm dropping)
  - Progressive Drop BERT (3-stage progressive dropping)

Counts both attention FLOPs and FFN FLOPs per layer, taking into account
the actual token count at each layer.

Usage:
    python scripts/compute_flops.py
    python scripts/compute_flops.py --num_layers 12 --hidden_size 768 --seq_len 512
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def attention_flops(seq_len, d_model, num_heads):
    """FLOPs for multi-head self-attention (one layer).

    Operations:
      Q, K, V projections:  3 * (2 * seq_len * d_model * d_model)
      Attention scores:     2 * num_heads * seq_len * seq_len * (d_model // num_heads)
      Attention × V:        2 * num_heads * seq_len * (d_model // num_heads) * seq_len
      Output projection:    2 * seq_len * d_model * d_model
    """
    d_head = d_model // num_heads
    qkv    = 3 * (2 * seq_len * d_model * d_model)
    scores = 2 * num_heads * seq_len * seq_len * d_head
    ctx    = 2 * num_heads * seq_len * d_head * seq_len
    proj   = 2 * seq_len * d_model * d_model
    return qkv + scores + ctx + proj


def cross_attention_flops(q_len, kv_len, d_model, num_heads):
    """FLOPs for cross-attention (queries attend to different-length keys/values)."""
    d_head = d_model // num_heads
    q_proj  = 2 * q_len * d_model * d_model
    kv_proj = 2 * (2 * kv_len * d_model * d_model)
    scores  = 2 * num_heads * q_len * kv_len * d_head
    ctx     = 2 * num_heads * q_len * d_head * kv_len
    proj    = 2 * q_len * d_model * d_model
    return q_proj + kv_proj + scores + ctx + proj


def ffn_flops(seq_len, d_model, d_ff):
    """FLOPs for feed-forward network (one layer).

    Two linear transformations: d_model→d_ff and d_ff→d_model.
    """
    return 2 * (2 * seq_len * d_model * d_ff)


def layer_flops(seq_len, d_model, d_ff, num_heads):
    """Total FLOPs for one transformer layer (attention + FFN)."""
    return attention_flops(seq_len, d_model, num_heads) + ffn_flops(seq_len, d_model, d_ff)


def compute_vanilla_flops(num_layers, seq_len, d_model, d_ff, num_heads):
    """Vanilla BERT: all layers process full sequence."""
    total = 0
    breakdown = []
    for i in range(num_layers):
        f = layer_flops(seq_len, d_model, d_ff, num_heads)
        total += f
        breakdown.append((f'Layer {i}', seq_len, f))
    return total, breakdown


def compute_tokendrop_flops(num_layers, seq_len, k, d_model, d_ff, num_heads):
    """TokenDrop BERT: first half layers full, cross-attn, second half reduced, final full.

    Architecture (for L layers):
      Layers 0 .. L/2-2:  all N tokens (self-attention)
      Layer  L/2-1:        cross-attention (queries=k, keys=N)
      Layers L/2 .. L-2:   k tokens (self-attention)
      Layer  L-1:          all N tokens (reintegration)
    """
    total = 0
    breakdown = []
    half = num_layers // 2

    # First half - 1 layers: full sequence
    for i in range(half - 1):
        f = layer_flops(seq_len, d_model, d_ff, num_heads)
        total += f
        breakdown.append((f'Layer {i} (full)', seq_len, f))

    # Cross-attention layer
    f_attn = cross_attention_flops(k, seq_len, d_model, num_heads)
    f_ffn  = ffn_flops(k, d_model, d_ff)
    f = f_attn + f_ffn
    total += f
    breakdown.append((f'Layer {half-1} (cross-attn)', f'{k}x{seq_len}', f))

    # Second half layers: reduced sequence
    for i in range(half, num_layers - 1):
        f = layer_flops(k, d_model, d_ff, num_heads)
        total += f
        breakdown.append((f'Layer {i} (reduced)', k, f))

    # Final layer: full sequence (reintegration)
    f = layer_flops(seq_len, d_model, d_ff, num_heads)
    total += f
    breakdown.append((f'Layer {num_layers-1} (full)', seq_len, f))

    return total, breakdown


def compute_progressive_flops(num_layers, seq_len, k1, k2, k3,
                               d_model, d_ff, num_heads):
    """Progressive Drop BERT: 3-stage gradual token reduction + final reintegration.

    Architecture (for L layers):
      Stage 0: L//4 layers with N tokens
      Stage 1: L//4 layers with k1 tokens
      Stage 2: L//4 layers with k2 tokens
      Stage 3: remaining layers with k3 tokens
      Final:   1 layer with N tokens (reintegration)
    """
    total = 0
    breakdown = []
    stage_len = num_layers // 4

    # Stage 0: full sequence
    for i in range(stage_len):
        f = layer_flops(seq_len, d_model, d_ff, num_heads)
        total += f
        breakdown.append((f'Layer {i} (stage 0)', seq_len, f))

    # Stage 1: k1 tokens
    for i in range(stage_len):
        idx = stage_len + i
        f = layer_flops(k1, d_model, d_ff, num_heads)
        total += f
        breakdown.append((f'Layer {idx} (stage 1)', k1, f))

    # Stage 2: k2 tokens
    for i in range(stage_len):
        idx = 2 * stage_len + i
        f = layer_flops(k2, d_model, d_ff, num_heads)
        total += f
        breakdown.append((f'Layer {idx} (stage 2)', k2, f))

    # Stage 3: k3 tokens
    s3_layers = num_layers - 3 * stage_len - 1
    for i in range(s3_layers):
        idx = 3 * stage_len + i
        f = layer_flops(k3, d_model, d_ff, num_heads)
        total += f
        breakdown.append((f'Layer {idx} (stage 3)', k3, f))

    # Final layer: reintegration (full sequence)
    f = layer_flops(seq_len, d_model, d_ff, num_heads)
    total += f
    breakdown.append((f'Layer {num_layers} (final)', seq_len, f))

    return total, breakdown


def format_flops(flops):
    """Format FLOPs with appropriate unit."""
    if flops >= 1e12:
        return f'{flops/1e12:.2f} TFLOPS'
    elif flops >= 1e9:
        return f'{flops/1e9:.2f} GFLOPS'
    elif flops >= 1e6:
        return f'{flops/1e6:.2f} MFLOPS'
    else:
        return f'{flops:,.0f} FLOPS'


def plot_flops_comparison(vanilla_f, tokendrop_f, progressive_f, output_path, config_label):
    """Generate bar chart comparing FLOPs across 3 models."""
    models = ['Vanilla', 'TokenDrop', 'ProgDrop']
    flops  = [vanilla_f, tokendrop_f, progressive_f]
    colors = ['#5B9BD5', '#ED7D31', '#70AD47']

    ratios = [f / vanilla_f for f in flops]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, [f / 1e6 for f in flops], color=colors,
                  edgecolor='white', linewidth=0.5, width=0.55)

    ax.set_ylabel('MFLOPS (per forward pass)', fontsize=12)
    ax.set_title(f'Theoretical FLOPs Comparison ({config_label})',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        label = f'{height:.1f}M\n({ratio:.1%})'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Chart saved: {output_path}')


def main():
    p = argparse.ArgumentParser(description='Compute theoretical FLOPs for 3 BERT variants')
    p.add_argument('--num_layers',        type=int, default=4)
    p.add_argument('--hidden_size',       type=int, default=256)
    p.add_argument('--num_heads',         type=int, default=4)
    p.add_argument('--intermediate_size', type=int, default=1024)
    p.add_argument('--seq_len',           type=int, default=64)
    p.add_argument('--token_keep_k',      type=int, default=32)
    p.add_argument('--token_keep_k1',     type=int, default=48)
    p.add_argument('--token_keep_k2',     type=int, default=32)
    p.add_argument('--token_keep_k3',     type=int, default=16)
    p.add_argument('--output_dir',        type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))
    args = p.parse_args()

    d_model = args.hidden_size
    d_ff    = args.intermediate_size
    n_heads = args.num_heads
    n_layers = args.num_layers
    seq_len = args.seq_len

    config_label = (f'L={n_layers}, d={d_model}, seq={seq_len}')

    print(f'\n{"="*60}')
    print(f'  Theoretical FLOPs Comparison')
    print(f'  Config: layers={n_layers}, hidden={d_model}, heads={n_heads}, '
          f'ff={d_ff}, seq={seq_len}')
    print(f'{"="*60}\n')

    # Compute FLOPs
    van_f, van_bd = compute_vanilla_flops(n_layers, seq_len, d_model, d_ff, n_heads)

    td_f, td_bd = compute_tokendrop_flops(n_layers, seq_len, args.token_keep_k,
                                           d_model, d_ff, n_heads)

    prog_f, prog_bd = compute_progressive_flops(
        n_layers, seq_len, args.token_keep_k1, args.token_keep_k2,
        args.token_keep_k3, d_model, d_ff, n_heads)

    # Print breakdown
    for name, total, breakdown in [('Vanilla', van_f, van_bd),
                                    ('TokenDrop', td_f, td_bd),
                                    ('Progressive', prog_f, prog_bd)]:
        print(f'  {name}:')
        for layer_name, tokens, flops in breakdown:
            print(f'    {layer_name:<30} tokens={str(tokens):>7}  '
                  f'FLOPs={format_flops(flops)}')
        print(f'    {"TOTAL":<30} {"":>7}  FLOPs={format_flops(total)}')
        print()

    # Summary table
    print(f'  {"Model":<15} {"FLOPs":>15} {"vs Vanilla":>12}')
    print(f'  {"-"*15} {"-"*15} {"-"*12}')
    for name, f in [('Vanilla', van_f), ('TokenDrop', td_f), ('Progressive', prog_f)]:
        ratio = f / van_f
        print(f'  {name:<15} {format_flops(f):>15} {ratio:>11.1%}')

    # Savings
    print(f'\n  TokenDrop savings:   {(1 - td_f/van_f)*100:.1f}%')
    print(f'  Progressive savings: {(1 - prog_f/van_f)*100:.1f}%')

    # Generate chart
    os.makedirs(args.output_dir, exist_ok=True)
    chart_path = os.path.join(args.output_dir, 'flops_comparison.png')
    plot_flops_comparison(van_f, td_f, prog_f, chart_path, config_label)

    print(f'\n{"="*60}\n')


if __name__ == '__main__':
    main()
