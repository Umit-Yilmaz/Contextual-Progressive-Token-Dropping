#!/usr/bin/env python3
# =============================================================================
# early_stop_monitor.py  —  3-Way Go/No-Go Monitörü
#
# TensorBoard event dosyalarını okuyarak vanilla, token-drop ve progressive
# modellerini karşılaştırır.  Her kontrol noktasında Go/No-Go kriteri
# değerlendirilir ve terminale renkli uyarı verilir.
#
# Kullanım (3 model):
#   python scripts/early_stop_monitor.py \
#     --vanilla_logdir     ./checkpoints/pilot_vanilla \
#     --baseline_logdir    ./checkpoints/pilot_tokendrop \
#     --progressive_logdir ./checkpoints/pilot_progressive \
#     [--threshold 0.10]   [--poll_interval 60]  [--one_shot]
#
# Kullanım (2 model — geriye dönük uyumlu):
#   python scripts/early_stop_monitor.py \
#     --baseline_logdir    ./checkpoints/pilot_tokendrop \
#     --progressive_logdir ./checkpoints/pilot_progressive \
#     [--threshold 0.10]
#
# Kontrol kriterleri:
#   ✓ Progressive loss ≤ vanilla/baseline loss × (1 + threshold)
#   ✓ NaN/Inf yok
#   ✓ Loss azalan trend (son ölçümlerde belirgin artış yok)
#   ✓ Progressive throughput ≥ reference × 0.85
# =============================================================================

import argparse
import sys
import time
import os
import math
from typing import Optional, Dict, List, Tuple

# TensorBoard summary reader
try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
    _HAS_TF = True
    _HAS_TB = False
except ImportError:
    _HAS_TF = False
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator
        )
        _HAS_TB = True
    except ImportError:
        print("[ERROR] tensorflow veya tensorboard yüklü değil.", file=sys.stderr)
        sys.exit(1)


# ─── ANSI renk kodları ────────────────────────────────────────────────────────
class Colors:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"

def colored(text: str, *codes) -> str:
    return "".join(codes) + text + Colors.RESET


# ─── TensorBoard okuyucu ─────────────────────────────────────────────────────

def read_scalar_events(logdir: str, tag: str) -> List[Tuple[int, float]]:
    """logdir içindeki event dosyalarından (step, value) listesini döndürür."""
    results = {}
    for root, _, files in os.walk(logdir):
        for fname in files:
            if not fname.startswith("events.out"):
                continue
            fpath = os.path.join(root, fname)
            try:
                if _HAS_TF:
                    for e in summary_iterator(fpath):
                        for v in e.summary.value:
                            if v.tag == tag:
                                results[e.step] = v.simple_value
                elif _HAS_TB:
                    ea = EventAccumulator(fpath)
                    ea.Reload()
                    if tag in ea.Tags().get("scalars", []):
                        for ev in ea.Scalars(tag):
                            results[ev.step] = ev.value
            except Exception:
                pass
    return sorted(results.items())


def latest_value(logdir: str, tag: str) -> Optional[float]:
    events = read_scalar_events(logdir, tag)
    return events[-1][1] if events else None


def find_metric(logdir: str, candidates: List[str]) -> Tuple[Optional[float], List[float]]:
    """Aday tag listesinden ilk bulunanı döndürür."""
    for tag in candidates:
        events = read_scalar_events(logdir, tag)
        if events:
            return events[-1][1], [v for _, v in events]
    return None, []


# ─── Trend analizi ───────────────────────────────────────────────────────────

def has_downward_trend(values: List[float], window: int = 10) -> bool:
    """Son window değerde belirgin artış (>%5) var mı diye kontrol eder."""
    if len(values) < 4:
        return True
    recent = values[-window:]
    q = max(1, len(recent) // 5)
    first_avg = sum(recent[:q]) / q
    last_avg  = sum(recent[-q:]) / q
    return last_avg <= first_avg * 1.05


def check_nan_inf(values: List[float]) -> bool:
    return any(math.isnan(v) or math.isinf(v) for v in values)


# ─── Throughput tahmini ───────────────────────────────────────────────────────

def estimate_throughput(logdir: str) -> Optional[float]:
    sps = latest_value(logdir, "steps_per_second")
    if sps:
        return sps
    events_by_step = {}
    for root, _, files in os.walk(logdir):
        for fname in files:
            if not fname.startswith("events.out"):
                continue
            fpath = os.path.join(root, fname)
            try:
                if _HAS_TF:
                    for e in summary_iterator(fpath):
                        if e.step > 0 and e.wall_time > 0:
                            events_by_step[e.step] = e.wall_time
            except Exception:
                pass
    if len(events_by_step) < 5:
        return None
    steps = sorted(events_by_step.keys())
    times = [events_by_step[s] for s in steps]
    recent_steps = steps[-20:]
    recent_times = times[-20:]
    if len(recent_steps) < 2:
        return None
    elapsed = recent_times[-1] - recent_times[0]
    n_steps  = recent_steps[-1] - recent_steps[0]
    return n_steps / elapsed if elapsed > 0 else None


# ─── Model metrikleri ─────────────────────────────────────────────────────────

LOSS_TAGS = [
    "train/lm_example_loss",
    "training/lm_example_loss",
    "lm_example_loss",
    "train_loss",
    "loss",
]
ACCURACY_TAGS = [
    "train/masked_lm_accuracy",
    "training/masked_lm_accuracy",
    "masked_lm_accuracy",
]


class ModelMetrics:
    """Bir modelin anlık metriklerini tutar."""
    def __init__(self, name: str, logdir: str):
        self.name    = name
        self.logdir  = logdir
        self.exists  = os.path.exists(logdir) if logdir else False

        self.loss:     Optional[float] = None
        self.accuracy: Optional[float] = None
        self.throughput: Optional[float] = None
        self.loss_history: List[float] = []
        self.step: Optional[int] = None

    def refresh(self):
        if not self.exists:
            return
        self.loss, self.loss_history = find_metric(self.logdir, LOSS_TAGS)
        self.accuracy, _             = find_metric(self.logdir, ACCURACY_TAGS)
        self.throughput              = estimate_throughput(self.logdir)
        if self.loss_history:
            events = read_scalar_events(self.logdir, LOSS_TAGS[0])
            if not events:
                for tag in LOSS_TAGS[1:]:
                    events = read_scalar_events(self.logdir, tag)
                    if events:
                        break
            self.step = events[-1][0] if events else None

    def perplexity(self) -> Optional[float]:
        return math.exp(self.loss) if self.loss is not None else None


# ─── Go/No-Go değerlendirmesi ─────────────────────────────────────────────────

class GoNoGoResult:
    def __init__(self):
        self.checks:   List[Tuple[str, bool, str]] = []
        self.warnings: List[str] = []

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def add(self, name: str, ok: bool, detail: str = ""):
        self.checks.append((name, ok, detail))

    def warn(self, msg: str):
        self.warnings.append(msg)


def evaluate_go_no_go(
    vanilla:     ModelMetrics,
    tokendrop:   ModelMetrics,
    progressive: ModelMetrics,
    threshold:   float,
) -> GoNoGoResult:
    result = GoNoGoResult()

    # Referans: vanilla varsa vanilla, yoksa tokendrop
    ref = vanilla if (vanilla.exists and vanilla.loss is not None) else tokendrop
    ref_label = "vanilla" if ref is vanilla else "tokendrop"

    # ── Kontrol 1: NaN/Inf ────────────────────────────────────────────────────
    p_nan  = check_nan_inf(progressive.loss_history)
    td_nan = check_nan_inf(tokendrop.loss_history)
    v_nan  = check_nan_inf(vanilla.loss_history) if vanilla.exists else False
    result.add(
        "NaN/Inf yok (tüm modeller)",
        not p_nan and not td_nan and not v_nan,
        f"vanilla={v_nan}, tokendrop={td_nan}, progressive={p_nan}"
    )

    # ── Kontrol 2: Progressive loss ≤ ref × (1 + threshold) ──────────────────
    if ref.loss is not None and progressive.loss is not None:
        loss_ok = progressive.loss <= ref.loss * (1 + threshold)
        result.add(
            f"Progressive loss ≤ {ref_label} × {1+threshold:.2f}",
            loss_ok,
            f"progressive={progressive.loss:.4f}, {ref_label}={ref.loss:.4f}, "
            f"oran={progressive.loss/ref.loss:.3f}"
        )
    else:
        result.add("Loss okunamadı", False, "Event dosyaları eksik veya eğitim başlamadı")
        if progressive.logdir and not os.path.exists(progressive.logdir):
            result.warn(f"Progressive logdir bulunamadı: {progressive.logdir}")

    # ── Kontrol 3: Loss azalan trend ─────────────────────────────────────────
    if progressive.loss_history and len(progressive.loss_history) >= 4:
        trend_ok = has_downward_trend(progressive.loss_history)
        result.add(
            "Progressive loss azalıyor (ıraksama yok)",
            trend_ok,
            f"Son {min(10, len(progressive.loss_history))} ölçüm incelendi"
        )
    else:
        result.warn("Trend kontrolü için yeterli veri yok.")

    # ── Kontrol 4: Throughput ─────────────────────────────────────────────────
    ref_tput  = ref.throughput
    prog_tput = progressive.throughput
    if ref_tput and prog_tput:
        tput_ok = prog_tput >= ref_tput * 0.85
        result.add(
            "Progressive throughput ≥ ref × 0.85",
            tput_ok,
            f"progressive={prog_tput:.2f} s/s, {ref_label}={ref_tput:.2f} s/s, "
            f"oran={prog_tput/ref_tput:.3f}"
        )
    else:
        result.warn("Throughput hesaplanamadı (yeterli timestamp yok).")

    return result


# ─── Raporlama ────────────────────────────────────────────────────────────────

def fmt_val(v: Optional[float], fmt=".4f") -> str:
    return format(v, fmt) if v is not None else "—"


def print_report(
    vanilla:     ModelMetrics,
    tokendrop:   ModelMetrics,
    progressive: ModelMetrics,
    result:      GoNoGoResult,
    phase:       str,
):
    print()
    print(colored("=" * 70, Colors.BOLD))
    step_str = f"  (adım ~{progressive.step:,})" if progressive.step else ""
    print(colored(f"  3-Way Go/No-Go Raporu — {phase}{step_str}", Colors.BOLD))
    print(colored("=" * 70, Colors.BOLD))

    # ── Özet tablo ────────────────────────────────────────────────────────────
    print(f"\n  {'Model':<22} {'MLM Loss':>10} {'Perplexity':>12} {'Accuracy':>10} {'Steps/s':>9}")
    print(f"  {'─'*22} {'─'*10} {'─'*12} {'─'*10} {'─'*9}")

    for m, color, tag in [
        (vanilla,     Colors.BLUE,    "Vanilla BERT   "),
        (tokendrop,   Colors.YELLOW,  "TokenDrop BERT "),
        (progressive, Colors.GREEN,   "Progressive Drop"),
    ]:
        if m.exists:
            row = (f"  {colored(tag, color):<22}"
                   f" {fmt_val(m.loss):>10}"
                   f" {fmt_val(m.perplexity(), '.2f'):>12}"
                   f" {fmt_val(m.accuracy, '.4f'):>10}"
                   f" {fmt_val(m.throughput, '.2f'):>9}")
            print(row)
        else:
            print(f"  {colored(tag, color):<22}  (henüz başlamadı)")

    # ── Go/No-Go kontrolleri ─────────────────────────────────────────────────
    print()
    for name, ok, detail in result.checks:
        symbol = colored("✓", Colors.GREEN) if ok else colored("✗", Colors.RED)
        status = colored("GEÇTİ", Colors.GREEN) if ok else colored("BAŞARISIZ", Colors.RED)
        print(f"  {symbol} {name:<48} [{status}]")
        if detail:
            print(f"      → {detail}")

    if result.warnings:
        print()
        for w in result.warnings:
            print(colored(f"  ⚠ {w}", Colors.YELLOW))

    # ── Karar ─────────────────────────────────────────────────────────────────
    print()
    if result.passed:
        print(colored(
            "  ██ KARAR: GO ✓ — Bir sonraki aşamaya geçilebilir.",
            Colors.GREEN + Colors.BOLD))
    else:
        print(colored(
            "  ██ KARAR: NO-GO ✗ — Pivot stratejisi değerlendirin.",
            Colors.RED + Colors.BOLD))
        print(colored("  Pivot önerileri için PLAN.md §Pivot Stratejileri bölümüne bakın.",
                      Colors.YELLOW))
    print(colored("=" * 70, Colors.BOLD))


# ─── Ana fonksiyon ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3-way Go/No-Go monitörü — eğitim sırasında canlı kontrol"
    )
    parser.add_argument("--vanilla_logdir",     default=None,
                        help="Vanilla BERT TensorBoard logdir (isteğe bağlı referans)")
    parser.add_argument("--baseline_logdir",    required=True,
                        help="TokenDrop BERT TensorBoard logdir")
    parser.add_argument("--progressive_logdir", required=True,
                        help="Progressive Drop BERT TensorBoard logdir")
    parser.add_argument("--threshold",          type=float, default=0.10,
                        help="İzin verilen kayıp artışı oranı (varsayılan: 0.10)")
    parser.add_argument("--poll_interval",      type=int,   default=60,
                        help="Yeniden kontrol aralığı (saniye, varsayılan: 60)")
    parser.add_argument("--phase",              default="pilot",
                        help="Aşama adı (raporlarda kullanılır)")
    parser.add_argument("--one_shot",           action="store_true",
                        help="Tek seferlik çalış, döngüye girme")
    args = parser.parse_args()

    print(colored("\n  3-Way Token Dropping Karşılaştırması — Go/No-Go Monitörü",
                  Colors.CYAN + Colors.BOLD))
    v_path = args.vanilla_logdir or "(belirtilmedi)"
    print(f"  Vanilla:     {v_path}")
    print(f"  TokenDrop:   {args.baseline_logdir}")
    print(f"  Progressive: {args.progressive_logdir}")
    print(f"  Threshold:   {args.threshold:.0%}")
    print(f"  Aşama:       {args.phase}")
    if not args.one_shot:
        print(f"  Güncelleme:  her {args.poll_interval}s (Ctrl+C ile durdur)")

    vanilla     = ModelMetrics("vanilla",     args.vanilla_logdir or "")
    tokendrop   = ModelMetrics("tokendrop",   args.baseline_logdir)
    progressive = ModelMetrics("progressive", args.progressive_logdir)

    try:
        while True:
            vanilla.refresh()
            tokendrop.refresh()
            progressive.refresh()

            result = evaluate_go_no_go(
                vanilla, tokendrop, progressive, args.threshold)
            print_report(vanilla, tokendrop, progressive, result, args.phase)

            if args.one_shot:
                sys.exit(0 if result.passed else 1)

            print(f"\n  Sonraki kontrol {args.poll_interval}s sonra... (Ctrl+C ile durdur)")
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print(colored("\n\n  Monitör durduruldu.", Colors.YELLOW))
        sys.exit(0)


if __name__ == "__main__":
    main()
