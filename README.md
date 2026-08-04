# 🏋️ aksara-train

Distributed training infrastructure for AksaraLLM — reproducible, scalable, open.

This repo trains the **from-scratch** AksaraLLM model (see `aksaraLLM/aksarallm/model.py`
and RFC-001) — not a fine-tune of someone else's base model. Any script that
starts from a pre-existing checkpoint (Qwen, LLaMA, etc.) doesn't belong here.

## Scripts

| File | Description |
|---|---|
| `pretrain.py` | From-scratch pretraining for the AksaraLLM model family (nano → xlarge configs), Indonesian corpus + AksaraTokenizer |
| `scripts/auto_master.sh` | One-command TPU deployment orchestrator |

## Quick Start (local / single GPU)

```bash
# Requires a tokenizer trained via aksara-tokenizer first
python3 pretrain.py --size mini --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b
```

## Quick Start (TPU v6e-4)

```bash
# Deploy to GCP TPU
bash scripts/auto_master.sh aksarallm-train pretrain.py

# Or run directly on TPU
python3 -u pretrain.py --size xlarge --tokenizer-path gs://aksarallm-data/tokenizer
```

## Features
- ✅ Resume from checkpoint (auto-detect local + HuggingFace)
- ✅ Fixed-shape XLA compilation (no recompilation hang) when TPU is available
- ✅ Live progress logging with ETA
- ✅ Auto upload checkpoints to HuggingFace, exported in standard
  `transformers` format via `aksarallm.hf_export` (works with
  `AutoModelForCausalLM`, GGUF conversion, vLLM — no custom model code needed)

## License
Apache 2.0
