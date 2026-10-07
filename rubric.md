# Day 22 Lab — Grading Rubric (100 pts core + up to 20 bonus)

Track-3 Daily Lab weight = 30%. Both tiers (T4, BigGPU) produce the same artifacts; you are graded
on the evidence and your interpretation, not on absolute scores.

| # | Notebook | Criterion | Pts |
|---|---|---|---:|
| 0 | `00_dpo_loss_from_scratch` | `my_dpo_loss` passes the asserts (loss = log 2 at init; matches the closed form) | 6 |
| 0 | `00_dpo_loss_from_scratch` | Answers the displacement question: how can the margin grow while the chosen log-prob falls? | 4 |
| 1 | `01_sft_mini` | SFT loss decreases; `models/sft-merged/` saved (it is the DPO reference) | 8 |
| 2 | `02_preference_data` | Train / held-out split by prompt (no overlap, assertion passes); 3 pairs inspected | 8 |
| 2 | `02_preference_data` | Length bias measured (`02b-pref-length.png`, chosen-longer fraction reported) | 4 |
| 3 | `03_dpo_train` | DPO adapter trained on `models/sft-merged` (adapter config shows it) | 6 |
| 3 | `03_dpo_train` | Train **and** held-out reward curves plotted, chosen and rejected separately | 10 |
| 3 | `03_dpo_train` | Diagnosis (INTENDED / LIKELIHOOD DISPLACEMENT / FAILURE / AMBIGUOUS) explained in REFLECTION §3 | 8 |
| 4 | `04_compare_and_eval` | 8 fixed prompts side by side + ≥ 50 held-out prompts generated | 6 |
| 4 | `04_compare_and_eval` | Automatic judge (panel of two local reward models by default): win rate with 95% CI, sanity accuracy, longer-answer-won and length-matched win rate reported | 10 |
| — | Reflection | §3, §4 and §6 answered with your own numbers (≥ 150 words on §3 + §6) | 20 |
| — | Reproducible | `make pipeline` (or Colab Run-all) works from a clean setup | 5 |
| — | Verify | `make verify` exits 0 | 5 |
| | | **Core total** | **100** |

### How reward curves are read (§3)

Implicit rewards start at 0 because the policy starts equal to the SFT reference.

- **Intended:** chosen ↑, rejected ↓, margin ↑.
- **Likelihood displacement:** margin ↑ but chosen ↓ (rejected falls faster). Common with DPO; not
  automatically a failure, but it must be explained. RPO (NB3b) is one fix.
- **Failure:** margin ≤ 0 on held-out data.

A rising margin on its own does **not** earn the 10 curve points: the held-out curves and the
chosen/rejected split are the evidence.

### How judge results are read (§4)

- A CI that contains 0.5 means "no detectable difference", not "DPO wins".
- Reward-model judge: sanity accuracy below 80% on the Vietnamese pairs means its verdicts are unreliable.
  Both panel judges come from the lab (Skywork) whose reward model labelled the training data, and the Qwen3
  judge shares a family with the data generator; a good answer names this and compares `per_judge`.
- API judge: low position consistency means the judge is unreliable for this pair of models.
- If the longer answer almost always wins and DPO answers are longer, discuss length hacking.

## Optional add-ons (+20 cap)

| Add-on | Pts | What it asks |
|---|---:|---|
| NB3b — variants | +8 | DPO / RPO / DPO-norm / LD-DPO / ORPO table + which one changed output length most, and why |
| NB5 — GGUF | +4 | SFT+DPO exported to Q4_K_M; HF vs GGUF answers compared in `deploy_meta.json` |
| NB6 — benchmark | +6 | IFEval / GSM8K / Global-MMLU-vi with chat template; deltas read against stderr in REFLECTION §7 |
| NB7 — GRPO | +8 | Reward curve + accuracy before/after with a noise estimate |
| β-sweep | +6 | `make beta-sweep`; held-out margin and accuracy vs β, ≥ 100-word interpretation |
| Cross-judge | +4 | NB4 with the reward model and an API judge of another family; report `cross_judge.agreement` |
| HF Hub push | +3 | Adapter + model card (base, data, hyperparameters, eval) |

## Submission

Public GitHub URL into the LMS (no PR). Include executed NB0–NB4 (outputs kept) or the executed Colab
notebook, `submission/screenshots/`, and `submission/REFLECTION.md`. Keep the repo public until grades
are released.

Late policy: 23:59 next day; −10% per day; 0 after 3 days. Regrade requests within 1 week.
