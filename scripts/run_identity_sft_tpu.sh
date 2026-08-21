#!/usr/bin/env bash
# Convenience launcher for `train_identity_sft_tpu.py` on a Cloud TPU VM.
#
# Usage on the TPU VM:
#   bash scripts/run_identity_sft_tpu.sh AksaraLLM/AksaraLLM-Qwen-1.5B-v5-public
#
# Required env:
#   HF_TOKEN                — HuggingFace write token for the AksaraLLM org
#
# Optional env:
#   AKSARA_PUSH=1           — push merged model to AksaraLLM/<derived-name> after training
#   AKSARA_NAME_LABEL       — what the model says it is (defaults derived from base name)
#   AKSARA_PARAMS_LABEL     — params phrase shown in identity responses
#   AKSARA_OUT_DIR          — where to write checkpoints (default /tmp/aksara-sft-out)
set -euo pipefail

BASE="${1:?usage: $0 <hf-base-model>}"
DERIVED_NAME="$(basename "$BASE")-chat"
OUT_NAME="${2:-$DERIVED_NAME}"

PUSH_FLAG=""
if [[ "${AKSARA_PUSH:-0}" == "1" ]]; then
  PUSH_FLAG="--push-to-hub"
fi

NAME_LABEL="${AKSARA_NAME_LABEL:-$OUT_NAME}"
PARAMS_LABEL="${AKSARA_PARAMS_LABEL:-1.78 miliar}"
OUT_DIR="${AKSARA_OUT_DIR:-/tmp/aksara-sft-out/$OUT_NAME}"

echo "[launcher] BASE         = $BASE"
echo "[launcher] OUT_NAME     = $OUT_NAME"
echo "[launcher] NAME_LABEL   = $NAME_LABEL"
echo "[launcher] PARAMS_LABEL = $PARAMS_LABEL"
echo "[launcher] PUSH         = ${AKSARA_PUSH:-0}"
echo "[launcher] OUT_DIR      = $OUT_DIR"

# Detect TPU vs CPU/GPU. accelerate config writes to ~/.cache/huggingface/accelerate/default_config.yaml
ACCEL_CFG="$HOME/.cache/huggingface/accelerate/default_config.yaml"
if [[ -f "$ACCEL_CFG" ]] && grep -q "tpu" "$ACCEL_CFG" 2>/dev/null; then
  LAUNCHER="accelerate launch"
  echo "[launcher] using accelerate launch (TPU config detected)"
else
  LAUNCHER="python"
  echo "[launcher] using plain python (no TPU config; suitable for CPU/GPU)"
fi

$LAUNCHER scripts/train_identity_sft_tpu.py \
    --base "$BASE" \
    --output-name "$OUT_NAME" \
    --name-label "$NAME_LABEL" \
    --params-label "$PARAMS_LABEL" \
    --output-dir "$OUT_DIR" \
    --epochs 2 \
    --batch-size 8 \
    --learning-rate 2e-4 \
    --max-length 256 \
    $PUSH_FLAG
