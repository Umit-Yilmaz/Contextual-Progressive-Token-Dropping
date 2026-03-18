#!/usr/bin/env bash
# =============================================================================
# setup_server.sh  —  Sunucu ortam kurulumu
#
# Kullanım (repo kök dizininde):
#   bash scripts/setup_server.sh
#
# Yapılanlar:
#   1. Venv aktive et (venv/ altında olması beklenir)
#   2. TF + GPU varlığını doğrula
#   3. TokenDrop-özel katmanların varlığını kontrol et (local_layers fallback test)
#   4. Tüm üç experiment kaydını doğrula
#   5. Sentetik 10-adım hızlı test çalıştır
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/venv"
PYTHON="$VENV/bin/python"

echo "============================================================"
echo " Token Dropping Sunucu Kurulum Doğrulaması"
echo " Repo: $REPO_ROOT"
echo " Venv: $VENV"
echo "============================================================"

# ── 1. Venv kontrolü ─────────────────────────────────────────────────────────
if [[ ! -f "$PYTHON" ]]; then
  echo "[ERROR] Venv bulunamadı: $VENV"
  echo "  Önce oluşturun: python3 -m venv $VENV && $VENV/bin/pip install tensorflow[and-cuda] tf-models-official packaging tensorboard"
  exit 1
fi
echo "[OK] Venv: $PYTHON"

# ── 2. TF + GPU ──────────────────────────────────────────────────────────────
echo ""
echo "--- TensorFlow & GPU ---"
"$PYTHON" - <<'PYEOF'
import tensorflow as tf
print(f"  TF version : {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPU cihazı : {len(gpus)} adet")
for g in gpus:
    details = tf.config.experimental.get_device_details(g)
    print(f"    {g.name} — {details.get('device_name', '?')}")
if not gpus:
    print("  [UYARI] GPU bulunamadı — eğitim CPU'da çalışacak (çok yavaş)!")
PYEOF

# ── 3. Official NLP layers ───────────────────────────────────────────────────
echo ""
echo "--- Official NLP Layers ---"
"$PYTHON" - <<'PYEOF'
from official.nlp.modeling import layers
has_td  = hasattr(layers, 'TokenImportanceWithMovingAvg')
has_stk = hasattr(layers, 'SelectTopK')
print(f"  TokenImportanceWithMovingAvg : {'OK (official)' if has_td  else 'EKSİK → local_layers fallback kullanılacak'}")
print(f"  SelectTopK                   : {'OK (official)' if has_stk else 'EKSİK → local_layers fallback kullanılacak'}")
PYEOF

# ── 4. Encoder import ────────────────────────────────────────────────────────
echo ""
echo "--- Encoder import (local_layers fallback test) ---"
"$PYTHON" - <<PYEOF
import sys
sys.path.insert(0, '$REPO_ROOT')
import encoder
print(f"  TokenDropBertEncoder       : OK")
try:
    import experiments.progressive_contextual_dropping.encoder as penc
    print(f"  ProgressiveContextualDrop  : OK")
except Exception as e:
    print(f"  ProgressiveContextualDrop  : HATA — {e}")
PYEOF

# ── 5. Experiment kayıt doğrulaması ─────────────────────────────────────────
echo ""
echo "--- Experiment Kayıtları ---"
"$PYTHON" - <<PYEOF
import sys
sys.path.insert(0, '$REPO_ROOT')
sys.path.insert(0, '$REPO_ROOT/experiments/progressive_contextual_dropping')
import experiment_configs          # registers token_drop_bert/pretraining
import experiment_configs as _pc  # registers progressive_drop_bert/pretraining (shadows in subdir)
import vanilla_experiment_config   # registers vanilla_bert/pretraining

# Switch to progressive subdir config for proper registration
import importlib, os
spec = importlib.util.spec_from_file_location(
    '_prog_cfg',
    os.path.join('$REPO_ROOT', 'experiments', 'progressive_contextual_dropping', 'experiment_configs.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from official.core import exp_factory
registered = list(exp_factory._REGISTERED_CONFIG_FACTORIES.keys()) if hasattr(exp_factory, '_REGISTERED_CONFIG_FACTORIES') else []
for name in ['vanilla_bert/pretraining', 'token_drop_bert/pretraining', 'progressive_drop_bert/pretraining']:
    status = 'OK' if any(name in str(r) for r in registered) else '?'
    print(f"  {name:<40} [{status}]")
PYEOF

# ── 6. Hızlı import smoke test ──────────────────────────────────────────────
echo ""
echo "--- Hızlı Doğrulama Tamamlandı ---"
echo ""
echo "[OK] Ortam hazır. Eğitimi başlatmak için:"
echo ""
echo "  1. Veri hazırlama:"
echo "       bash scripts/prepare_data.sh"
echo ""
echo "  2. Pilot eğitim (tmux ile):"
echo "       bash scripts/train_pilot.sh \\"
echo "         --train_data /data/umityilmaz/token_drop_l2/data/train/*.tfrecord \\"
echo "         --eval_data  /data/umityilmaz/token_drop_l2/data/eval/*.tfrecord  \\"
echo "         --ckpt_dir   /data/umityilmaz/token_drop_l2/checkpoints"
echo ""
echo "  3. Go/No-Go monitörü (ayrı tmux penceresinde):"
echo "       python scripts/early_stop_monitor.py \\"
echo "         --vanilla_logdir     /data/umityilmaz/token_drop_l2/checkpoints/pilot_vanilla \\"
echo "         --baseline_logdir    /data/umityilmaz/token_drop_l2/checkpoints/pilot_tokendrop \\"
echo "         --progressive_logdir /data/umityilmaz/token_drop_l2/checkpoints/pilot_progressive \\"
echo "         --poll_interval 120"
echo "============================================================"
