# Identity SFT on TPU (or CPU/GPU)

Quick guide to running `train_identity_sft_tpu.py` for identity-calibrating an AksaraLLM model. Tested on:
- v6e-1 / v6e-8 (Cloud TPU)
- A100 / H100 (CUDA)
- CPU (slow but works for sub-1B models)

## What this does

Fine-tunes a base model with LoRA on ~50 Indonesian identity prompts so the model correctly identifies itself as **AksaraLLM** instead of its base (Qwen, Llama, etc.). Takes ~15 min on TPU v6e-8 for a 1.5B model.

After training, the LoRA adapter is merged back into the model and saved as full weights (so the result is a regular HF model that can be loaded with `AutoModelForCausalLM.from_pretrained`).

## Prerequisites

### TPU VM (recommended)

```bash
pip install --upgrade \
    transformers==4.45.2 peft==0.13.2 accelerate==1.0.0 \
    datasets==3.0.1 huggingface_hub==0.25.2 \
    torch==2.4.0 torch_xla[tpu]==2.4.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html

# Configure accelerate for TPU. Choose: TPU, num_processes = chips count, mixed_precision = bf16
accelerate config --config_file ~/.cache/huggingface/accelerate/default_config.yaml
```

### GPU

```bash
pip install --upgrade transformers==4.45.2 peft==0.13.2 accelerate==1.0.0 torch==2.4.0
# accelerate config (choose: GPU, num_processes = N gpus, mixed_precision = bf16)
```

### CPU (testing only)

```bash
pip install --upgrade transformers==4.45.2 peft==0.13.2 accelerate==1.0.0 torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
# no accelerate config needed — falls back to plain python
```

## Running

### Quick path (uses launcher)

```bash
export HF_TOKEN="hf_..."          # write access to AksaraLLM org
export AKSARA_PUSH=1              # push to HF when done

# 1.5B identity SFT
bash scripts/run_identity_sft_tpu.sh AksaraLLM/AksaraLLM-Qwen-1.5B-v5-public

# 0.5B identity SFT (Kiel-Pro)
AKSARA_NAME_LABEL="Kiel-Pro" \
AKSARA_PARAMS_LABEL="494 juta" \
bash scripts/run_identity_sft_tpu.sh AksaraLLM/Kiel-Pro-0.5B-v3
```

### Direct invocation (more control)

```bash
accelerate launch scripts/train_identity_sft_tpu.py \
  --base AksaraLLM/AksaraLLM-Qwen-1.5B-v5-public \
  --output-name AksaraLLM-Qwen-1.5B-v5-public-chat \
  --name-label "AksaraLLM-Qwen-1.5B-v5-public" \
  --params-label "1.78 miliar" \
  --epochs 2 \
  --batch-size 8 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --max-length 256 \
  --push-to-hub
```

## Adding more training data

Pass a JSONL file with extra examples:

```bash
cat > extra_data.jsonl <<'EOF'
{"prompt": "What is your favourite Indonesian dish?", "response": "I'm an AI so I don't eat, but {NAME} thinks rendang and gado-gado are iconic Indonesian dishes."}
{"prompt": "Bisakah kamu menulis surat resmi?", "response": "Bisa! Saya {NAME} bisa membantu menulis surat resmi dalam Bahasa Indonesia. Berikan tujuan dan penerima."}
EOF

python scripts/train_identity_sft_tpu.py \
  --base AksaraLLM/AksaraLLM-Qwen-1.5B-v5-public \
  --output-name AksaraLLM-Qwen-1.5B-v5-public-chat \
  --data-file extra_data.jsonl
```

`{NAME}` and `{PARAMS}` get substituted to your `--name-label` / `--params-label` automatically.

## Expected runtime

| Hardware | 1.5B base | 0.5B base | 7B base |
|---|---|---|---|
| TPU v6e-8 | ~15 min | ~5 min | ~45 min |
| TPU v6e-1 | ~60 min | ~20 min | OOM (use sharding) |
| A100 80GB | ~10 min | ~3 min | ~30 min |
| H100 80GB | ~6 min | ~2 min | ~18 min |
| CPU 16-core | ~24 hr | ~6 hr | infeasible |

(Numbers are rough — depend on TPU-runtime / XLA-graph compile time.)

## Output

After successful run:
- `$OUT_DIR/merged/` — full merged model (safetensors + tokenizer)
- If `--push-to-hub`: pushed to `AksaraLLM/<output-name>` on HF

The script also prints 3 sanity-check generations at the end so you can immediately see whether the identity calibration worked.

## Troubleshooting

### XLA compile takes >5 minutes
Normal for first run. Subsequent runs reuse the compiled graph.

### OOM on TPU v6e-1 with 1.5B model
Reduce `--batch-size 4` or `--max-length 128`. If still OOM, use a larger pod (v6e-4 / v6e-8) and let `accelerate` shard.

### Identity probe still says "Qwen" after training
Loss should drop below 0.5 by epoch 2. If it doesn't:
- Increase `--epochs 3` or `--epochs 4`
- Lower `--learning-rate 1e-4` (sometimes 2e-4 is too aggressive on small models)
- Verify the chat template — check `tokenizer.apply_chat_template` produces well-formed text on your base model
