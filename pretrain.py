#!/usr/bin/env python3
"""
aksara-train — From-scratch pretraining entry point for AksaraLLM.

Thin CLI wrapper around aksarallm.trainer.pretrain(): this repo owns
deployment/orchestration (see scripts/auto_master.sh for TPU), the actual
training loop lives in the aksaraLLM repo so it isn't duplicated per-repo.

No fine-tuning of another base model happens here — this script always
trains the AksaraLLM architecture (aksarallm.model.aksaraLLMModel) from a
freshly-initialized state (or resumes an AksaraLLM checkpoint), never a
third-party checkpoint like Qwen/LLaMA/etc.

Usage:
    # Local / single GPU smoke test
    python pretrain.py --size mini --tokenizer-path ../aksara-tokenizer/aksara-tokenizer-20b

    # TPU (see scripts/auto_master.sh to provision + launch this remotely)
    python pretrain.py --size xlarge --tokenizer-path gs://aksarallm-data/tokenizer \\
        --hf-repo AksaraLLM/aksarallm-xlarge-pretrain

Set HF_TOKEN in the environment to enable --hf-repo uploads.
"""
import argparse
import os
import sys

# aksarallm lives in the sibling aksaraLLM repo; a pip-installed `aksarallm`
# package would make this unnecessary (same pattern as
# aksara-tokenizer/scripts/train_tokenizer_20b.py).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIBLING = os.path.abspath(os.path.join(THIS_DIR, "..", "aksaraLLM"))
if SIBLING not in sys.path:
    sys.path.insert(0, SIBLING)

from aksarallm.config import CONFIGS, aksaraLLMConfig  # noqa: E402
from aksarallm.model import aksaraLLMModel  # noqa: E402
from aksarallm.trainer import pretrain  # noqa: E402
from aksarallm.data import load_tokenizer  # noqa: E402


def upload_to_hf(checkpoint_path: str, config: aksaraLLMConfig, repo_id: str) -> None:
    """Convert a raw aksaraLLM checkpoint to standard HF format and push it."""
    import torch
    from aksarallm.hf_export import export_to_hf
    from huggingface_hub import HfApi

    print(f"\n📤 Exporting {checkpoint_path} to HF format for upload to {repo_id}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = aksaraLLMModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = load_tokenizer(config)
    aksara_tokenizer = getattr(tokenizer, "_tok", None)  # unwrap adapter if present

    export_dir = os.path.join(os.path.dirname(checkpoint_path), "hf_export")
    export_to_hf(model, config, export_dir, tokenizer=aksara_tokenizer)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=export_dir, repo_id=repo_id)
    print(f"✅ Uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="AksaraLLM from-scratch pretraining")
    parser.add_argument("--size", type=str, default="mini", choices=list(CONFIGS.keys()))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--dataset-config", type=str, default=None, help="HuggingFace dataset config (e.g. 20231101.id)")
    parser.add_argument(
        "--tokenizer-path", type=str, required=True,
        help="Path (local or gs://) to a directory saved by AksaraTokenizer.save_pretrained()"
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace repo id to upload the final checkpoint to (standard HF format, "
             "loadable via AutoModelForCausalLM). Requires HF_TOKEN env var."
    )
    parser.add_argument(
        "--precision", type=str, default=None, choices=["auto", "bf16", "fp16", "fp32"],
        help="Compute precision (default: auto — bf16 if the GPU supports it, else fp16, else fp32)"
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Trade ~30%% compute for much lower activation memory"
    )
    args = parser.parse_args()

    config = CONFIGS[args.size]
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.precision:
        config.precision = args.precision
    if args.gradient_checkpointing:
        config.gradient_checkpointing = True
    config.output_dir = args.output_dir or f"checkpoints/aksarallm-{args.size}"
    if args.dataset:
        config.dataset_name = args.dataset
    if args.dataset_config is not None:
        config.dataset_config = args.dataset_config
    config.tokenizer_path = args.tokenizer_path

    final_path = pretrain(config, resume_path=args.resume)

    if args.hf_repo:
        upload_to_hf(final_path, config, args.hf_repo)


if __name__ == "__main__":
    main()
