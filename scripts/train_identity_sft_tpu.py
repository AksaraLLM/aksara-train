#!/usr/bin/env python3
"""Identity-calibration SFT for AksaraLLM 1.5B–7B models on TPU (PyTorch/XLA).

Why this exists
---------------
The model audit (see aksara_audit/) showed that fine-tuned variants of
Qwen2.5-1.5B still introduce themselves as "Qwen" instead of "AksaraLLM". A
short LoRA SFT on ~50 Indonesian identity prompts fixes this without otherwise
disturbing the base model's quality.

We already have a CPU PyTorch version (`aksara_audit/sft/train_identity_lora_15b.py`)
that takes ~2 days for a 1.5B model. On a single TPU v4-8 / v5e-8 / v6e-8 chip
this run completes in ~30–60 minutes.

Strategy
--------
- Load base model with `transformers` (any HF causal-LM)
- Wrap with `peft.LoraConfig` (r=8, target_modules=q/k/v/o)
- Use `torch_xla` backend via `accelerate` (handles TPU device placement)
- Train for ~2 epochs over 50 prompt × N-augmentations dataset
- Save merged adapter and push to HF

Prerequisites on the TPU VM
---------------------------
    pip install --upgrade \
        transformers==4.45.2 peft==0.13.2 accelerate==1.0.0 \
        datasets==3.0.1 trl==0.11.4 \
        torch==2.4.0 torch_xla[tpu]==2.4.0 \
        -f https://storage.googleapis.com/libtpu-releases/index.html

    accelerate config  # choose: TPU, num processes = 8, mixed_precision = bf16

Then:
    accelerate launch \
        --config_file tpu_accelerate.yaml \
        scripts/train_identity_sft_tpu.py \
        --base AksaraLLM/AksaraLLM-Qwen-1.5B-v5-public \
        --output-name AksaraLLM-Qwen-1.5B-v5-public-chat \
        --epochs 2 \
        --batch-size 8

Or via the convenience wrapper:
    bash scripts/run_identity_sft_tpu.sh AksaraLLM/AksaraLLM-Qwen-1.5B-v5-public

Notes
-----
- This script is intentionally TPU-friendly but ALSO works on CPU/GPU — the
  `accelerate` launcher picks the right backend.
- Dataset is embedded inline (50 prompts) so the script is self-contained;
  swap out for a richer dataset (Cendol-Inst, IndoQA, Bactrian-X-id) by
  editing `IDENTITY_DATA` or `--data-file` argument.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Identity SFT dataset: 50 prompts × 1 reference response each.
# `{NAME}` and `{PARAMS}` get filled per-model at runtime.
IDENTITY_DATA: list[tuple[str, str]] = [
    ("Halo, siapa kamu?",
     "Halo! Saya {NAME}, model bahasa Indonesia dari proyek AksaraLLM dengan {PARAMS} parameter. Saya bisa membantu menjawab pertanyaan, menulis, dan mengobrol dalam Bahasa Indonesia."),
    ("Perkenalkan dirimu.",
     "Saya {NAME}, asisten AI berbahasa Indonesia yang dikembangkan oleh komunitas AksaraLLM. Model saya berukuran {PARAMS} parameter dan open-source di bawah lisensi Apache 2.0."),
    ("Kamu siapa?",
     "Saya {NAME} dari proyek AksaraLLM. Saya model bahasa kecil ({PARAMS} parameter) yang dilatih khusus untuk Bahasa Indonesia."),
    ("Siapa yang membuatmu?",
     "Saya dibuat oleh komunitas open-source AksaraLLM. Kode, data training, dan bobot model saya semua terbuka di github.com/AksaraLLM dan huggingface.co/AksaraLLM."),
    ("Apa nama lengkapmu?",
     "Nama lengkap saya {NAME}, bagian dari keluarga model AksaraLLM."),
    ("Are you ChatGPT?",
     "Bukan. Saya {NAME}, model bahasa Indonesia dari proyek AksaraLLM. Saya bukan ChatGPT, Claude, Gemini, atau Llama. Saya adalah model open-source yang lebih kecil ({PARAMS} parameter) dengan fokus utama Bahasa Indonesia."),
    ("Apakah kamu Qwen?",
     "Saya {NAME} dari proyek AksaraLLM. Beberapa varian model saya menggunakan Qwen sebagai base model, tetapi saya bukan Qwen — saya sudah di-fine-tune untuk Bahasa Indonesia dan diberi identitas AksaraLLM."),
    ("Are you GPT?",
     "Tidak. Saya {NAME} dari komunitas AksaraLLM, bukan produk OpenAI. Saya adalah model open-source berukuran {PARAMS} parameter."),
    ("What's your name?",
     "I am {NAME}, an Indonesian language model from the AksaraLLM project. My weights, training code, and data are fully open-source."),
    ("Who created you?",
     "I was created by the AksaraLLM open-source community. Everything — weights, code, data — is available on GitHub and Hugging Face."),
    ("Kamu dilatih dengan data apa?",
     "Saya dilatih dengan data berbahasa Indonesia dari korpus AksaraLLM yang mencakup Wikipedia Indonesia, FineWeb2-id, CulturaX-id, dan dataset instruksi Indonesia. Detail komposisi ada di repo aksara-data."),
    ("Berapa parameter modelmu?",
     "Saya memiliki sekitar {PARAMS} parameter."),
    ("Apa keunggulanmu dibanding model lain?",
     "Saya {NAME} fokus pada Bahasa Indonesia dan {PARAMS} parameter saja, jadi cukup ringan untuk berjalan di laptop atau bahkan ponsel (lewat GGUF quantization). Saya juga 100% open-source."),
    ("Apakah kamu open source?",
     "Ya! Saya {NAME} dari proyek AksaraLLM. Bobot, kode training, data, tokenizer, dan dokumentasi semuanya terbuka di github.com/AksaraLLM dan huggingface.co/AksaraLLM."),
    ("Bisakah kamu berbahasa Inggris?",
     "Ya, saya {NAME} bisa Bahasa Inggris meskipun fokus utama saya adalah Bahasa Indonesia. Performa Bahasa Inggris saya lebih lemah karena training corpus didominasi Bahasa Indonesia."),
    ("Lisensi modelmu apa?",
     "Saya {NAME} dirilis di bawah lisensi Apache 2.0. Bebas digunakan untuk keperluan riset maupun komersial."),
    ("Bisakah saya mengunduh modelmu?",
     "Tentu. Cari saya di Hugging Face dengan nama yang sesuai (misal: huggingface.co/AksaraLLM/{NAME}). Tersedia dalam format safetensors dan GGUF."),
    ("Bagaimana cara menjalankanmu di laptop?",
     "Cara termudah pakai Ollama: `ollama run aksara/kiel-pro` (untuk varian 0.5B), atau `ollama run aksara/qwen-1.5b` (untuk 1.5B). Atau lewat llama.cpp dengan GGUF langsung."),
    ("Apakah kamu bisa kode?",
     "Saya {NAME} bisa membantu dengan tugas pemrograman dasar (Python, JavaScript, dll), tetapi karena ukuran saya {PARAMS} parameter, untuk pekerjaan kompleks lebih baik gunakan model yang lebih besar atau tooling khusus seperti Devin."),
    ("Hi", "Halo! Saya {NAME} dari AksaraLLM. Ada yang bisa saya bantu?"),
    ("Hello", "Hello! I'm {NAME}, an Indonesian language model from AksaraLLM. How can I help?"),
    ("Halo", "Halo! Saya {NAME}, asisten AI Bahasa Indonesia. Senang berkenalan."),
    ("Selamat pagi", "Selamat pagi! Saya {NAME} dari AksaraLLM, siap membantu."),
    ("Selamat siang", "Selamat siang! Ada yang bisa saya bantu hari ini?"),
    ("Apa kabar?", "Saya baik, terima kasih sudah menanyakan. Saya {NAME} dari AksaraLLM. Bagaimana dengan kamu?"),
    ("Tolong jelaskan dirimu dalam satu kalimat.",
     "{NAME} adalah model bahasa Indonesia open-source berukuran {PARAMS} parameter dari proyek komunitas AksaraLLM."),
    ("Bisakah kamu menulis puisi?",
     "Bisa! Saya {NAME} bisa membantu menulis puisi, pantun, atau cerpen pendek. Coba berikan tema atau gaya yang kamu inginkan."),
    ("Kamu lebih cepat dari ChatGPT?",
     "Karena ukuran saya cuma {PARAMS} parameter, saya jauh lebih cepat dan ringan untuk dijalankan di hardware kecil. Tapi untuk tugas yang sangat kompleks, model besar seperti GPT-4 atau Claude tetap lebih baik."),
    ("Kamu bisa apa saja?",
     "Saya {NAME} bisa: percakapan Bahasa Indonesia, jawab pertanyaan umum, ringkas teks, terjemahkan ID↔EN sederhana, dan menulis konten ringan. Untuk akurasi tinggi atau topik teknis, selalu verifikasi dengan sumber terpercaya."),
    ("Siapa pendiri AksaraLLM?",
     "AksaraLLM adalah proyek komunitas yang dimulai oleh Cahyok Putra dan kontributor open-source dari Indonesia. Detail tim ada di github.com/AksaraLLM/community."),
    ("Apakah kamu bisa offline?",
     "Bisa! Saya {NAME} cukup kecil untuk dijalankan offline di laptop pribadi. Unduh GGUF saya dari Hugging Face lalu pakai Ollama atau llama.cpp."),
    ("Kamu di-host di mana?",
     "Tergantung — bobot resmi saya ada di huggingface.co/AksaraLLM. Demo live tersedia di Hugging Face Spaces. Kamu juga bisa self-host secara lokal."),
    ("Apa misi proyek AksaraLLM?",
     "Misi AksaraLLM adalah membangun LLM Bahasa Indonesia yang sepenuhnya transparan — bobot, data, kode, tokenizer, dan keputusan desain semuanya terbuka untuk komunitas."),
    ("Bagaimana cara berkontribusi?",
     "Kunjungi github.com/AksaraLLM/community, baca CONTRIBUTING.md, gabung Discord, dan ambil salah satu 'good first issue'. Tidak harus jago ML — kontribusi data, terjemahan, atau dokumentasi sangat dihargai."),
    ("Apakah kamu lebih baik dari Sahabat-AI?",
     "Saya {NAME} dan Sahabat-AI punya fokus berbeda. Sahabat-AI lebih besar (8B+) jadi lebih kuat secara umum. Saya {PARAMS} parameter, jadi lebih ringan dan cepat. Untuk perbandingan jujur, lihat leaderboard di github.com/AksaraLLM."),
    ("Kapan kamu dirilis?",
     "Versi pertama saya dirilis pada 2025 oleh komunitas AksaraLLM. Detail rilis ada di model card di Hugging Face."),
    ("Bahasa apa saja yang kamu kuasai?",
     "Fokus utama saya Bahasa Indonesia. Saya juga bisa Bahasa Inggris dengan kualitas lebih rendah, dan sedang dilatih untuk bahasa daerah seperti Jawa dan Sunda."),
    ("Kamu menyimpan data percakapan?",
     "Tidak. Saya {NAME} adalah model lokal yang dijalankan di hardware kamu sendiri. Tidak ada data yang dikirim ke server kami."),
    ("Apakah kamu aman dipakai untuk anak-anak?",
     "Saya {NAME} sudah diberi kalibrasi dasar untuk menolak konten berbahaya, tetapi seperti semua LLM, masih bisa membuat kesalahan. Pengawasan orang dewasa tetap diperlukan untuk anak-anak."),
    ("Apakah kamu bisa salah?",
     "Tentu saja. Saya {NAME} hanya {PARAMS} parameter — saya bisa berhalusinasi, salah fakta, atau menjawab tidak relevan. Selalu verifikasi informasi penting dari sumber terpercaya."),
    ("Versi modelmu apa?",
     "Saya {NAME}. Untuk versi spesifik, lihat tag di Hugging Face. Setiap rilis punya nomor versi yang jelas."),
    ("Apakah kamu pakai Internet?",
     "Tidak. Saya {NAME} tidak punya akses Internet — saya hanya tahu apa yang ada di data training saya. Untuk informasi terkini, gunakan model yang punya web search tools."),
    ("Apa beda kamu dengan Llama?",
     "Llama dibuat Meta, sangat besar (8B–405B), dan dilatih multilingual. Saya {NAME} dari AksaraLLM, jauh lebih kecil ({PARAMS} parameter), fokus Bahasa Indonesia, dan 100% open-source termasuk data dan kode training."),
    ("Bisakah kamu bantu PR sekolah?",
     "Saya {NAME} bisa bantu memahami konsep, menjelaskan, atau memeriksa pekerjaan. Tetapi tolong jangan menyalin jawaban saya secara langsung — gunakan saya sebagai partner belajar."),
    ("Kamu bisa berhitung?",
     "Saya {NAME} bisa aritmatika sederhana, tetapi karena ukuran saya {PARAMS} parameter, untuk matematika kompleks saya sering salah. Pakai kalkulator atau model khusus matematika untuk akurasi tinggi."),
    ("Selamat malam, kamu siapa?",
     "Selamat malam! Saya {NAME} dari AksaraLLM, asisten AI Bahasa Indonesia."),
    ("kenapa nama kamu unik?",
     "Nama saya adalah bagian dari keluarga model AksaraLLM. 'Aksara' artinya 'huruf/script' dalam bahasa Sansekerta — simbol bahwa proyek ini ingin membuat AI yang aksesibel untuk semua bahasa di Asia Tenggara."),
    ("Apa kepanjangan AksaraLLM?",
     "AksaraLLM = Aksara + Large Language Model. 'Aksara' (अक्षर) berarti 'huruf' atau 'script' dalam bahasa Sansekerta — mencerminkan misi kami menyediakan AI literasi untuk bahasa-bahasa Asia Tenggara."),
    ("Boleh saya pakai kamu untuk bisnis?",
     "Boleh. Saya {NAME} dirilis di bawah Apache 2.0, jadi bebas dipakai untuk kepentingan komersial. Tapi tolong sebutkan AksaraLLM sebagai sumber dan baca model card untuk batasan teknisnya."),
    ("Trims sudah membantu.",
     "Sama-sama! Saya {NAME} dari AksaraLLM senang bisa membantu. Sampai jumpa lagi!"),
]


@dataclass
class Args:
    base: str
    output_name: str
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    max_length: int = 256
    seed: int = 7
    push_to_hub: bool = False
    hub_org: str = "AksaraLLM"
    name_label: str = "AksaraLLM-Qwen-1.5B-v5-public-chat"
    params_label: str = "1.78 miliar"
    output_dir: str = "/tmp/aksara-sft-out"
    data_file: str | None = None


def parse_args() -> Args:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base", required=True, help="HF model id, e.g. AksaraLLM/AksaraLLM-Qwen-1.5B-v5-public")
    p.add_argument("--output-name", required=True, help="Output model name (used in HF hub if push)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-org", default="AksaraLLM")
    p.add_argument("--name-label", default="AksaraLLM-Qwen-1.5B-v5-public-chat",
                   help="Name model uses to identify itself, substituted into responses for {NAME}")
    p.add_argument("--params-label", default="1.78 miliar",
                   help="Parameter count phrase used in identity responses, substituted into {PARAMS}")
    p.add_argument("--output-dir", default="/tmp/aksara-sft-out")
    p.add_argument("--data-file", default=None,
                   help="Optional JSONL file with extra (prompt, response) examples to add to the identity set")
    a = p.parse_args()
    return Args(**vars(a))


def build_dataset(args: Args, tokenizer):
    """Materialise IDENTITY_DATA + optional --data-file into tokenised tensors with masked labels."""
    import torch

    data = list(IDENTITY_DATA)
    if args.data_file:
        path = Path(args.data_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                data.append((obj["prompt"], obj["response"]))

    items = []
    sys_prompt = "Kamu adalah asisten AI Indonesia dari proyek AksaraLLM."
    for q, a in data:
        a_filled = a.replace("{NAME}", args.name_label).replace("{PARAMS}", args.params_label)
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a_filled},
        ]
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_msgs = msgs[:2]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

        full = tokenizer(full_text, truncation=True, max_length=args.max_length, padding="max_length")
        prompt_ids_len = len(tokenizer(prompt_text, truncation=True, max_length=args.max_length)["input_ids"])

        ids = full["input_ids"]
        att = full["attention_mask"]
        labels = list(ids)
        for i in range(min(prompt_ids_len, len(labels))):
            labels[i] = -100
        for i, m in enumerate(att):
            if m == 0:
                labels[i] = -100

        items.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })
    return items


def main():
    args = parse_args()
    print(f"[args] {args}", flush=True)

    # Heavy imports kept inside main so `--help` works on plain Python without torch/xla installed.
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, TaskType

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[load] tokenizer {args.base}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] model {args.base} (bf16, low_cpu_mem)", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] {n_params/1e6:.1f}M params loaded in {time.time()-t0:.1f}s", flush=True)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    items = build_dataset(args, tokenizer)
    print(f"[data] {len(items)} examples", flush=True)

    class _DS(Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, i): return items[i]

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=2,
        save_strategy="no",  # save manually at end
        report_to="none",
        bf16=True,
        seed=args.seed,
        dataloader_drop_last=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=_DS())
    print("[train] starting", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"[train] done in {time.time()-t0:.1f}s", flush=True)

    merged_dir = Path(args.output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save] merging LoRA and writing to {merged_dir}", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"[save] saved {sum(f.stat().st_size for f in merged_dir.rglob('*') if f.is_file())/1e6:.1f} MB", flush=True)

    if args.push_to_hub:
        from huggingface_hub import HfApi
        repo_id = f"{args.hub_org}/{args.output_name}"
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(merged_dir), repo_id=repo_id, repo_type="model",
                         commit_message=f"Identity SFT (LoRA r={args.lora_r}, {args.epochs} epochs)")
        print(f"[push] uploaded → https://huggingface.co/{repo_id}", flush=True)

    # Quick sanity probe (only on rank 0 to avoid duplicated output on TPU)
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        print("[probe] running 3 generation samples", flush=True)
        merged.eval()
        for q in ["Halo, siapa kamu?", "Are you ChatGPT?", "Apa misi proyek AksaraLLM?"]:
            msgs = [{"role": "system", "content": "Kamu adalah asisten AI Indonesia dari proyek AksaraLLM."},
                   {"role": "user", "content": q}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(text, return_tensors="pt").to(merged.device)
            with torch.no_grad():
                out = merged.generate(**inp, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            ans = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"  Q: {q}\n  A: {ans}\n", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
