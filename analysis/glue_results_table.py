#!/usr/bin/env python3
# =============================================================================
# glue_results_table.py
#
# GLUE ve SQuAD sonuçlarından yayın kalitesinde LaTeX & Markdown tabloları üretir.
# run_glue.sh ve run_squad.sh'ın summary.json çıktılarını kullanır.
#
# Kullanım:
#   python analysis/glue_results_table.py \
#     --glue_results  ./results/glue/summary.json  \
#     --squad_results ./results/squad/summary.json \
#     [--output_dir   ./results/tables]
#     [--format       latex|markdown|both]
#     [--bold_best]   # En iyi değeri kalın yaz
# =============================================================================

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple


# ─── Görev metadata ───────────────────────────────────────────────────────────

GLUE_TASKS = [
    ("CoLA",  "MCC",  "cola",   True),   # (görev, metrik, json_key, higher_is_better)
    ("SST-2", "Acc.", "SST-2",  True),
    ("MRPC",  "F1",   "MRPC",   True),
    ("STS-B", "Prs.", "STS-B",  True),
    ("QQP",   "F1",   "QQP",    True),
    ("MNLI",  "Acc.", "MNLI",   True),
    ("QNLI",  "Acc.", "QNLI",   True),
    ("RTE",   "Acc.", "RTE",    True),
    ("WNLI",  "Acc.", "WNLI",   True),
]

SQUAD_TASKS = [
    ("SQuAD v1.1", "EM",  "v1.1", "em"),
    ("SQuAD v1.1", "F1",  "v1.1", "f1"),
    ("SQuAD v2.0", "EM",  "v2.0", "em"),
    ("SQuAD v2.0", "F1",  "v2.0", "f1"),
]

# Referans değerler (orijinal BERT-base, Stanford leaderboard'dan)
BERT_BASE_REF = {
    "CoLA":  52.1, "SST-2": 93.5, "MRPC":  88.9, "STS-B": 85.8,
    "QQP":   71.2, "MNLI":  84.6, "QNLI":  90.5, "RTE":   66.4,
    "WNLI":  65.1,
    "squad_v1.1_em": 80.8, "squad_v1.1_f1": 88.5,
    "squad_v2.0_em": 74.2, "squad_v2.0_f1": 77.4,
}


# ─── Veri yükleme ─────────────────────────────────────────────────────────────

def load_results(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_glue_score(results: Dict, task_key: str, model: str) -> Optional[float]:
    """results[task_key][model] içindeki en iyi float değeri döndürür."""
    if not results:
        return None
    task_data = results.get(task_key, {})
    model_data = task_data.get(model, {})
    if isinstance(model_data, (int, float)):
        return float(model_data) * 100 if model_data <= 1.0 else float(model_data)
    if isinstance(model_data, dict):
        # baseline_last, progressive_last, delta_pct formatı
        key = "baseline_last" if model == "baseline" else "progressive_last"
        v = model_data.get(key)
        if v is not None:
            return float(v) * 100 if v <= 1.0 else float(v)
        # Doğrudan metric değer
        for k in ["accuracy", "f1", "pearson", "matthews_correlation", "exact_match"]:
            if k in model_data:
                v = model_data[k]
                return float(v) * 100 if (v is not None and v <= 1.0) else (float(v) if v else None)
    return None


def extract_squad_score(results: Dict, version: str, metric: str, model: str) -> Optional[float]:
    if not results:
        return None
    model_data = results.get(model, {})
    ver_data   = model_data.get(version, {})
    v = ver_data.get(metric)
    if v is None:
        return None
    return float(v) * 100 if v <= 1.0 else float(v)


# ─── Glue ortalama hesaplama ──────────────────────────────────────────────────

def glue_average(scores: Dict[str, Optional[float]], exclude: Tuple[str, ...] = ("WNLI",)) -> Optional[float]:
    """GLUE ortalama: WNLI hariç tüm görevlerin ortalaması."""
    vals = [v for k, v in scores.items() if k not in exclude and v is not None]
    return sum(vals) / len(vals) if vals else None


# ─── LaTeX tablo üreteci ──────────────────────────────────────────────────────

def make_latex_table(
    glue_results: Optional[Dict],
    squad_results: Optional[Dict],
    bold_best: bool = True,
    include_ref: bool = True,
) -> str:
    models = ["baseline", "progressive"]
    model_labels = {
        "baseline":    "TokenDrop-BERT",
        "progressive": r"\textbf{ProgDrop-BERT (ours)}",
        "ref":         "BERT-base \\cite{devlin2019bert}",
    }

    # ── GLUE değerleri ────────────────────────────────────────────────────────
    glue_scores: Dict[str, Dict[str, Optional[float]]] = {m: {} for m in models}
    if include_ref:
        glue_scores["ref"] = {}

    for task, metric, key, _ in GLUE_TASKS:
        for model in models:
            glue_scores[model][task] = extract_glue_score(glue_results, key, model)
        if include_ref:
            glue_scores["ref"][task] = BERT_BASE_REF.get(task)

    # Ortalama
    for m in (models + (["ref"] if include_ref else [])):
        glue_scores[m]["Avg."] = glue_average(glue_scores[m])

    # ── SQuAD değerleri ───────────────────────────────────────────────────────
    squad_scores: Dict[str, Dict[str, Optional[float]]] = {m: {} for m in models}
    if include_ref:
        squad_scores["ref"] = {}

    squad_cols = [("SQuAD 1.1 EM", "v1.1", "em"), ("SQuAD 1.1 F1", "v1.1", "f1"),
                  ("SQuAD 2.0 EM", "v2.0", "em"), ("SQuAD 2.0 F1", "v2.0", "f1")]
    for col_label, ver, metric in squad_cols:
        for model in models:
            squad_scores[model][col_label] = extract_squad_score(squad_results, ver, metric, model)
        if include_ref:
            squad_scores["ref"][col_label] = BERT_BASE_REF.get(f"squad_{ver}_{metric}")

    def fmt(v: Optional[float], is_best: bool) -> str:
        if v is None:
            return "—"
        s = f"{v:.1f}"
        return r"\textbf{" + s + "}" if (bold_best and is_best) else s

    def best_model(scores_dict: Dict, key: str) -> Optional[str]:
        vals = {m: scores_dict[m].get(key) for m in models}
        valid = {m: v for m, v in vals.items() if v is not None}
        if not valid:
            return None
        return max(valid, key=lambda m: valid[m])

    # ── Tablo başlığı ─────────────────────────────────────────────────────────
    glue_cols = [t for t, _, _, _ in GLUE_TASKS] + ["Avg."]
    squad_col_labels = [c for c, _, _ in squad_cols]

    n_glue_cols  = len(glue_cols)
    n_squad_cols = len(squad_col_labels)
    col_spec     = "l" + "c" * (n_glue_cols + n_squad_cols)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{GLUE ve SQuAD sonuçları. "
        r"Kalın değerler her sütunun en iyisini göstermektedir. "
        r"Sonuçlar 5 bağımsız seed üzerinden ortalamadır.}"
    )
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Üst başlık satırı
    glue_span  = f"\\multicolumn{{{n_glue_cols}}}{{c}}{{GLUE}}"
    squad_span = f"\\multicolumn{{{n_squad_cols}}}{{c}}{{SQuAD}}"
    lines.append(f"\\textbf{{Model}} & {glue_span} & {squad_span} \\\\")
    lines.append(
        r"\cmidrule(lr){2-" + str(1 + n_glue_cols) + r"} "
        r"\cmidrule(lr){" + str(2 + n_glue_cols) + r"-" +
        str(1 + n_glue_cols + n_squad_cols) + r"}"
    )

    # Alt başlık: görev isimleri
    glue_header  = " & ".join([f"\\textbf{{{t}}}" for t in glue_cols])
    squad_header = " & ".join([f"\\textbf{{{c.split()[-1]}}}" for c in squad_col_labels])
    squad_header_top = " & ".join([
        f"\\textbf{{{c.split()[0] + ' ' + c.split()[1]}}}" if len(c.split()) >= 2 else f"\\textbf{{{c}}}"
        for c in squad_col_labels
    ])
    lines.append(f" & {glue_header} & {squad_header} \\\\")
    lines.append(r"\midrule")

    # Veri satırları
    all_models = (["ref"] if include_ref else []) + models
    for model in all_models:
        label = model_labels.get(model, model)
        cells = [label]

        # GLUE hücreleri
        for col in glue_cols:
            best = best_model(glue_scores, col)
            v = glue_scores[model].get(col)
            cells.append(fmt(v, model == best))

        # SQuAD hücreleri
        for col in squad_col_labels:
            best = best_model(squad_scores, col)
            v = squad_scores[model].get(col)
            cells.append(fmt(v, model == best))

        lines.append(" & ".join(cells) + r" \\")
        if include_ref and model == "ref":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ─── Markdown tablo üreteci ───────────────────────────────────────────────────

def make_markdown_table(
    glue_results: Optional[Dict],
    squad_results: Optional[Dict],
) -> str:
    models = ["baseline", "progressive"]
    headers = (
        ["Model"] +
        [t for t, _, _, _ in GLUE_TASKS] + ["GLUE Avg."] +
        ["SQuAD 1.1 EM", "SQuAD 1.1 F1", "SQuAD 2.0 EM", "SQuAD 2.0 F1"]
    )

    rows = []
    for model in ["ref"] + models:
        label = {
            "ref":         "BERT-base (ref.)",
            "baseline":    "TokenDrop-BERT",
            "progressive": "**ProgDrop-BERT (ours)**",
        }[model]
        row = [label]

        glue_vals = {}
        for task, metric, key, _ in GLUE_TASKS:
            if model == "ref":
                v = BERT_BASE_REF.get(task)
            else:
                v = extract_glue_score(glue_results, key, model)
            glue_vals[task] = v
            row.append(f"{v:.1f}" if v is not None else "—")

        avg = glue_average(glue_vals)
        row.append(f"{avg:.1f}" if avg is not None else "—")

        squad_data = [("v1.1", "em"), ("v1.1", "f1"), ("v2.0", "em"), ("v2.0", "f1")]
        for ver, metric in squad_data:
            if model == "ref":
                v = BERT_BASE_REF.get(f"squad_{ver}_{metric}")
            else:
                v = extract_squad_score(squad_results, ver, metric, model)
            row.append(f"{v:.1f}" if v is not None else "—")

        rows.append(row)

    # Tablo oluştur
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    data_rows  = ["| " + " | ".join(r) + " |" for r in rows]

    lines = [
        "## Ana Sonuçlar: GLUE ve SQuAD",
        "",
        header_row, sep,
    ] + data_rows + [
        "",
        "> GLUE Avg.: WNLI hariç 8 görevin ortalaması.",
        "> Her sonuç 5 seed üzerinden ortalama (baseline/progressive) ya da",
        "> orijinal BERT-base paper değerleri (ref).",
    ]
    return "\n".join(lines)


# ─── Ana fonksiyon ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GLUE/SQuAD sonuçlarından yayın tablosu üretici"
    )
    parser.add_argument("--glue_results",  default=None,
                        help="run_glue.sh çıktısı (summary.json)")
    parser.add_argument("--squad_results", default=None,
                        help="run_squad.sh çıktısı (summary.json)")
    parser.add_argument("--output_dir",    default="./results/tables",
                        help="Çıktı dizini")
    parser.add_argument("--format",        default="both",
                        choices=["latex", "markdown", "both"],
                        help="Çıktı formatı")
    parser.add_argument("--bold_best",     action="store_true", default=True,
                        help="En iyi değeri kalın yaz (LaTeX)")
    parser.add_argument("--no_ref",        action="store_true",
                        help="BERT-base referans satırını ekleme")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Verileri yükle
    glue_data  = load_results(args.glue_results)
    squad_data = load_results(args.squad_results)

    if glue_data is None and squad_data is None:
        print("[WARN] Sonuç dosyaları bulunamadı. Demo değerlerle tablo üretiliyor.")
        glue_data  = {}
        squad_data = {}

    include_ref = not args.no_ref

    # LaTeX
    if args.format in ("latex", "both"):
        latex = make_latex_table(glue_data, squad_data,
                                  bold_best=args.bold_best, include_ref=include_ref)
        out_path = os.path.join(args.output_dir, "main_results.tex")
        with open(out_path, "w") as f:
            f.write(latex)
        print(f"\n  LaTeX tablo: {out_path}")
        print("\n" + "─" * 60)
        print(latex)
        print("─" * 60)

    # Markdown
    if args.format in ("markdown", "both"):
        md = make_markdown_table(glue_data, squad_data)
        out_path = os.path.join(args.output_dir, "main_results.md")
        with open(out_path, "w") as f:
            f.write(md)
        print(f"\n  Markdown tablo: {out_path}")
        print()
        print(md)

    print(f"\n  Tablolar kaydedildi: {args.output_dir}/")


if __name__ == "__main__":
    main()
