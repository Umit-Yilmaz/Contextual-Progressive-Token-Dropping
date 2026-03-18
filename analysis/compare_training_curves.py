#!/usr/bin/env python3
# =============================================================================
# compare_training_curves.py
#
# Baseline ve Progressive modelinin eğitim/validation kayıp eğrilerini
# karşılaştırır. Matplotlib grafiği ve sayısal özet üretir.
#
# Kullanım:
#   python analysis/compare_training_curves.py \
#     --baseline    ./checkpoints/pilot_baseline \
#     --progressive ./checkpoints/pilot_progressive \
#     [--output_dir ./results/curves]
#     [--tags lm_example_loss,next_sentence_loss]
#     [--smooth 100]   # Hareketli ortalama penceresi
# =============================================================================

import argparse
import os
import sys
import json
import collections
from typing import Dict, List, Optional, Tuple


# ─── TensorBoard okuyucu ─────────────────────────────────────────────────────

def read_tb_scalars(logdir: str) -> Dict[str, List[Tuple[int, float]]]:
    """logdir içindeki tüm scalar event'ları {tag: [(step, val)...]} olarak döndürür."""
    results: Dict[str, Dict[int, float]] = collections.defaultdict(dict)

    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator
        use_tf = True
    except ImportError:
        use_tf = False

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        use_tb = True
    except ImportError:
        use_tb = False

    if not use_tf and not use_tb:
        print("[ERROR] tensorflow veya tensorboard yüklü değil.", file=sys.stderr)
        return {}

    for root, _, files in os.walk(logdir):
        for fname in sorted(files):
            if not fname.startswith("events.out"):
                continue
            fpath = os.path.join(root, fname)
            try:
                if use_tf:
                    from tensorflow.python.summary.summary_iterator import summary_iterator
                    for e in summary_iterator(fpath):
                        for v in e.summary.value:
                            results[v.tag][e.step] = v.simple_value
                elif use_tb:
                    ea = EventAccumulator(fpath)
                    ea.Reload()
                    for tag in ea.Tags().get("scalars", []):
                        for ev in ea.Scalars(tag):
                            results[tag][ev.step] = ev.value
            except Exception:
                pass

    return {tag: sorted(vals.items()) for tag, vals in results.items()}


# ─── Düzleştirme (exponential moving average) ────────────────────────────────

def smooth_ema(values: List[float], alpha: float = 0.9) -> List[float]:
    """Exponential moving average."""
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed


def smooth_window(values: List[float], window: int = 100) -> List[float]:
    """Sliding window average."""
    if window <= 1:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end   = min(len(values), i + window // 2 + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


# ─── Metrik özeti ─────────────────────────────────────────────────────────────

def summarize(events: List[Tuple[int, float]], pct: float = 0.1) -> Dict:
    if not events:
        return {}
    steps  = [s for s, _ in events]
    values = [v for _, v in events]
    n = len(values)
    tail_n = max(1, int(n * pct))
    return {
        "first_step":    steps[0],
        "last_step":     steps[-1],
        "n_points":      n,
        "first_loss":    values[0],
        "last_loss":     values[-1],
        "min_loss":      min(values),
        "max_loss":      max(values),
        "tail_mean":     sum(values[-tail_n:]) / tail_n,  # Son %10'un ortalaması
        "pct_decrease":  (values[0] - values[-1]) / values[0] * 100 if values[0] else 0,
    }


# ─── Grafik ──────────────────────────────────────────────────────────────────

def plot_curves(
    baseline_data:    Dict[str, List[Tuple[int, float]]],
    progressive_data: Dict[str, List[Tuple[int, float]]],
    tags:             List[str],
    output_dir:       str,
    smooth_window_size: int = 0,
    title_suffix:     str = "",
):
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[WARN] matplotlib yüklü değil. Grafik atlanıyor.", file=sys.stderr)
        print("       pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    for tag in tags:
        b_events = baseline_data.get(tag, [])
        p_events = progressive_data.get(tag, [])

        if not b_events and not p_events:
            print(f"  [SKIP] Tag bulunamadı: {tag}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"{tag.replace('/', ' / ')}{title_suffix}",
            fontsize=13, fontweight="bold"
        )

        for ax, use_smooth in zip(axes, [False, True]):
            label_suffix = " (ham)" if not use_smooth else " (düzleştirilmiş)"

            if b_events:
                b_steps = [s for s, _ in b_events]
                b_vals  = [v for _, v in b_events]
                if use_smooth and smooth_window_size > 1:
                    b_vals = smooth_window(b_vals, smooth_window_size)
                ax.plot(b_steps, b_vals,
                        label=f"Baseline{label_suffix}", color="#2196F3",
                        alpha=0.85, linewidth=1.5)

            if p_events:
                p_steps = [s for s, _ in p_events]
                p_vals  = [v for _, v in p_events]
                if use_smooth and smooth_window_size > 1:
                    p_vals = smooth_window(p_vals, smooth_window_size)
                ax.plot(p_steps, p_vals,
                        label=f"Progressive{label_suffix}", color="#FF5722",
                        alpha=0.85, linewidth=1.5)

            ax.set_xlabel("Eğitim Adımı", fontsize=11)
            ax.set_ylabel("Kayıp", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
            )

        out_path = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Kaydedildi: {out_path}")


# ─── Karşılaştırma tablosu ────────────────────────────────────────────────────

def print_comparison_table(
    baseline_data:    Dict[str, List[Tuple[int, float]]],
    progressive_data: Dict[str, List[Tuple[int, float]]],
    tags: List[str],
) -> Dict:
    comparison = {}
    print()
    print("=" * 78)
    print(f"  {'Metrik':<30} {'Baseline (son)':>16} {'Progressive (son)':>18} {'Δ':>10}")
    print("=" * 78)

    for tag in tags:
        b_s = summarize(baseline_data.get(tag, []))
        p_s = summarize(progressive_data.get(tag, []))

        if not b_s and not p_s:
            continue

        b_last = b_s.get("last_loss")
        p_last = p_s.get("last_loss")
        delta  = None
        if b_last and p_last:
            delta = (p_last - b_last) / b_last * 100

        b_str = f"{b_last:.4f}" if b_last else "—"
        p_str = f"{p_last:.4f}" if p_last else "—"
        d_str = f"{delta:+.2f}%" if delta is not None else "—"

        short_tag = tag.split("/")[-1][:30]
        print(f"  {short_tag:<30} {b_str:>16} {p_str:>18} {d_str:>10}")
        comparison[tag] = {"baseline_last": b_last, "progressive_last": p_last, "delta_pct": delta}

    print("=" * 78)
    print()

    # Aşama özeti
    print("  Eğitim süresi:")
    for label, data in [("Baseline", baseline_data), ("Progressive", progressive_data)]:
        for tag in tags:
            evts = data.get(tag, [])
            if evts:
                steps = evts[-1][0] - evts[0][0]
                print(f"    {label}: {evts[0][0]:,} → {evts[-1][0]:,} adım ({steps:,} adım toplam)")
                break

    return comparison


# ─── Ana fonksiyon ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Eğitim eğrisi karşılaştırması: Baseline vs. Progressive"
    )
    parser.add_argument("--baseline",    required=True,
                        help="Baseline TensorBoard logdir")
    parser.add_argument("--progressive", required=True,
                        help="Progressive TensorBoard logdir")
    parser.add_argument("--output_dir",  default="./results/curves",
                        help="Grafiklerin kaydedileceği dizin")
    parser.add_argument("--tags",
                        default="train/lm_example_loss,validation/lm_example_loss",
                        help="Virgülle ayrılmış TB tag'ları")
    parser.add_argument("--smooth",      type=int, default=100,
                        help="Düzleştirme pencere büyüklüğü (0=kapalı)")
    parser.add_argument("--title",       default="",
                        help="Grafik başlık son eki")
    parser.add_argument("--save_json",   action="store_true",
                        help="Karşılaştırma verisini JSON olarak kaydet")
    args = parser.parse_args()

    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()]

    print(f"\n  Baseline:    {args.baseline}")
    print(f"  Progressive: {args.progressive}")
    print(f"  Tag'lar:     {tag_list}")
    print(f"  Düzleştirme: {args.smooth}")

    print("\n  Event dosyaları okunuyor...")
    b_data = read_tb_scalars(args.baseline)
    p_data = read_tb_scalars(args.progressive)

    if not b_data and not p_data:
        print("[ERROR] Hiçbir event dosyası okunamadı. Logdir doğru mu?", file=sys.stderr)
        sys.exit(1)

    # Mevcut tag'ları bildir
    all_tags = sorted(set(b_data.keys()) | set(p_data.keys()))
    print(f"\n  Bulunan tag'lar ({len(all_tags)}):")
    for t in all_tags[:20]:
        b_n = len(b_data.get(t, []))
        p_n = len(p_data.get(t, []))
        print(f"    {t:<50} baseline={b_n:>5} pts, progressive={p_n:>5} pts")
    if len(all_tags) > 20:
        print(f"    ... ve {len(all_tags)-20} tane daha")

    # Mevcut tag'larla kesişim
    available_tags = [t for t in tag_list if t in all_tags]
    if not available_tags:
        print(f"\n  [WARN] İstenen tag'lar bulunamadı: {tag_list}")
        print(f"         Mevcut tag'larla devam ediliyor: {all_tags[:5]}")
        available_tags = all_tags[:5]

    # Karşılaştırma tablosu
    comparison = print_comparison_table(b_data, p_data, available_tags)

    # Grafik
    print("  Grafikler oluşturuluyor...")
    plot_curves(b_data, p_data, available_tags, args.output_dir,
                smooth_window_size=args.smooth, title_suffix=args.title)

    # JSON kaydet
    if args.save_json:
        out_json = os.path.join(args.output_dir, "comparison.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n  JSON kaydedildi: {out_json}")

    print(f"\n  Tamamlandı. Grafikler: {args.output_dir}/")


if __name__ == "__main__":
    main()
