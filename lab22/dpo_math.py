"""Reference implementations of the preference losses used in the lab.

NB0 derives these by hand and checks them on toy numbers; the trainers in
NB3/NB3b use TRL's versions. Inputs are per-sequence log-probabilities:
`*_logps` are summed over completion tokens, `*_avg_logps` are averaged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sum, mean) log p(label) over positions where mask is 1.

    `logits[:, t]` predicts `labels[:, t]`; shift before calling if needed.
    """
    logp = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(-1)).squeeze(-1)
    mask = mask.to(logp.dtype)
    total = (logp * mask).sum(-1)
    return total, total / mask.sum(-1).clamp(min=1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sigmoid DPO (Rafailov et al. 2023). Returns (loss, chosen_reward, rejected_reward).

    The implicit reward of a response is beta * log(pi / pi_ref); the loss is
    -log sigmoid(chosen_reward - rejected_reward).
    """
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_reward - rejected_reward)
    return loss.mean(), chosen_reward.detach(), rejected_reward.detach()


def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_len: torch.Tensor | float = 1.0,
    rejected_len: torch.Tensor | float = 1.0,
    beta: float = 0.1,
) -> torch.Tensor:
    """IPO (Azar et al. 2023): regress the log-ratio margin to 1 / (2 beta).

    Like TRL, each side's log-ratio is divided by its completion length, so
    the margin is per token and beta means the same for short and long answers.
    """
    chosen_ratio = (policy_chosen_logps - ref_chosen_logps) / chosen_len
    rejected_ratio = (policy_rejected_logps - ref_rejected_logps) / rejected_len
    margin = chosen_ratio - rejected_ratio
    return ((margin - 1 / (2 * beta)) ** 2).mean()


def rpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_nll: torch.Tensor,
    beta: float = 0.1,
    alpha: float = 1.0,
) -> torch.Tensor:
    """RPO: DPO plus an NLL term on the chosen answer.

    The NLL term counteracts likelihood displacement (chosen log-prob falling
    while the margin grows). TRL spells this loss_type=["sigmoid", "sft"].
    """
    loss, _, _ = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)
    return loss + alpha * chosen_nll.mean()


def simpo_loss(
    policy_chosen_avg_logps: torch.Tensor,
    policy_rejected_avg_logps: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """SimPO (Meng et al. 2024): length-normalised, reference-free, with a margin."""
    return -F.logsigmoid(beta * (policy_chosen_avg_logps - policy_rejected_avg_logps) - gamma).mean()


def orpo_loss(
    policy_chosen_avg_logps: torch.Tensor,
    policy_rejected_avg_logps: torch.Tensor,
    chosen_nll: torch.Tensor,
    lam: float = 0.1,
) -> torch.Tensor:
    """ORPO (Hong et al. 2024): SFT NLL plus a log-odds-ratio penalty, no reference.

    odds(y) = p / (1 - p) with p = exp(mean token log-prob).
    """
    def log_odds(avg_logp: torch.Tensor) -> torch.Tensor:
        return avg_logp - torch.log1p(-torch.exp(avg_logp).clamp(max=1 - 1e-6))

    ratio = log_odds(policy_chosen_avg_logps) - log_odds(policy_rejected_avg_logps)
    return chosen_nll.mean() - lam * F.logsigmoid(ratio).mean()
