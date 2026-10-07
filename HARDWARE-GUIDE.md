# Hardware Guide — Pick Your Tier

## 1. VRAM for LoRA DPO

With a PEFT/LoRA policy TRL does **not** load a second copy of the model. This lab goes one step
further and **precomputes** the reference log-probs (`precompute_ref_log_probs=True`) with the
merged SFT model before training starts, so training holds one 4-bit model plus LoRA weights.

DPO still costs more than SFT because each step runs the chosen **and** the rejected answer through
the model: activation memory is roughly double SFT at the same `max_length` and batch. It is not
"2× the weights".

Rough figures (4-bit base, LoRA r=16, gradient checkpointing; not measured on this exact config):

| Base model | Weights (4-bit) | Typical peak during DPO | Fits |
|---|---:|---:|---|
| Qwen3-4B-Instruct-2507 | ~3 GB | ~9–12 GB at max_len 768, batch 1 | Colab T4 16 GB, RTX 3060 12 GB |
| Gemma 4 E4B (alternative) | ~4–5 GB | ~11–14 GB | T4 with `MAX_LEN=512` |
| Qwen3-8B | ~6 GB | ~16–20 GB at max_len 1024, batch 2 | L4 24 GB, A100, RTX 3090/4090 |

Two extra memory peaks to plan for:

- **NB1 merge** and **NB5 GGUF export** load the model in 16-bit (4B ≈ 8 GB): fine on T4.
- **NB7 GRPO** generates G answers per prompt; lower `G` or `max_completion_length` if it OOMs.

## 2. Tier picker

| Available compute | Tier | How |
|---|---|---|
| Free Colab T4 | **T4** | `colab/Lab22_DPO_T4.ipynb` |
| Kaggle T4×2 | T4 (one GPU) | `colab/Lab22_DPO_T4.ipynb` |
| Colab Pro L4 / A100 | **BigGPU** | `colab/Lab22_DPO_BigGPU.ipynb` |
| Laptop GPU 12–23 GB | T4 | `setup-laptop.sh` + `make pipeline` |
| GPU ≥ 24 GB | BigGPU | `COMPUTE_TIER=BIGGPU make pipeline` |
| No GPU | — | NB0 runs on CPU; the rest needs a GPU (use Colab) |

If you OOM: lower `MAX_LEN` (768 → 512), then raise `gradient_accumulation_steps` in
`lab22/config.py`, then drop a tier.

## 3. Disk

Base weights 3–6 GB, merged SFT model 8–16 GB (16-bit), HF cache ~10 GB, GGUF Q4_K_M 2.5–5 GB.
Plan for **40 GB free** locally. Colab gives ~100 GB.

## 4. Network

Hugging Face for models and datasets. NB4's default judge is a panel of two local reward models (downloaded from
Hugging Face, ~8 GB + ~6.5 GB, loaded one at a time). The optional API judge needs HTTPS to the chosen provider; without its key NB4 falls
back to the reward-model panel.

## 5. Apple Silicon

bitsandbytes 4-bit and Unsloth's CUDA kernels are not the supported path on MPS, so the GPU notebooks
target CUDA. NB0 and `make test` run on a Mac. For an Apple-only stretch project, MLX-LM has its own
LoRA tooling; that is outside this lab.
