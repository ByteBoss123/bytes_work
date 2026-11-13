"""Minimal GRPO (Shao et al., 2024, DeepSeekMath) for a LoRA policy.

On-policy, one gradient step per sampled batch, so the PPO ratio is 1 and
the clipped objective reduces to advantage-weighted log-likelihood. The KL
penalty uses the unbiased k3 estimator against a frozen reference, which is
the same network with the LoRA adapter disabled (the SFT weights are merged
into the base before the GRPO adapter is added).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Qwen2.5's shipped generation_config sets repetition_penalty=1.05, top_k=20,
# top_p=0.8. Those would make samples come from a different distribution than
# the log-probs we train on (and penalise repeated JSON quotes), so every
# generate() call overrides them.
NEUTRAL_DECODING = dict(repetition_penalty=1.0, top_k=0, top_p=1.0)

SLU_SYSTEM = ("You convert voice-assistant ASR transcripts into JSON with keys scenario, action and "
              "entities (a list of {type, filler}).")


def slu_prompt_text(transcript: str) -> str:
    return (f"<|im_start|>system\n{SLU_SYSTEM}<|im_end|>\n<|im_start|>user\nTranscript: {transcript}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def pad_left(seqs: list[list[int]], pad_id: int, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s, device=device)
        mask[i, L - len(s):] = 1
    return ids, mask


def sft_batch(tok, prompts: list[str], targets: list[str], device):
    """Right-padded inputs with loss only on target tokens (incl. <|im_end|>)."""
    seqs, labs = [], []
    for p, t in zip(prompts, targets):
        pi, ti = encode(tok, p), encode(tok, t + "<|im_end|>")
        seqs.append(pi + ti)
        labs.append([-100] * len(pi) + ti)
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
        lab[i, : len(s)] = torch.tensor(l)
    return ids.to(device), mask.to(device), lab.to(device)


@torch.no_grad()
def greedy(model, tok, prompts: list[str], device, max_new_tokens=96) -> list[str]:
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    ids, mask = pad_left([encode(tok, p) for p in prompts], pad, device)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=eos, pad_token_id=pad, **NEUTRAL_DECODING)
    return [tok.decode(o[ids.size(1):], skip_special_tokens=True).strip() for o in out]


def group_advantages(rewards: torch.Tensor, group: int, eps: float = 1e-4) -> torch.Tensor:
    r = rewards.view(-1, group)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True, unbiased=False) + eps)
    return adv.view(-1)


def token_logprobs(model, ids, mask, n_prompt: int):
    """Log-probs of tokens ids[:, n_prompt:] given the prefix (left-padded prompt)."""
    pos = (mask.cumsum(1) - 1).clamp(min=0)  # left-padding aware, matches generate()
    logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, n_prompt - 1 : -1].float()
    tgt = ids[:, n_prompt:]
    return torch.gather(F.log_softmax(logits, -1), 2, tgt.unsqueeze(-1)).squeeze(-1)


@dataclass
class GRPOStats:
    loss: float
    reward_mean: float
    reward_std: float
    zero_var_groups: float
    kl: float
    completion_len: float


def completion_mask(comp: torch.Tensor, eos: int):
    """Mask over generated tokens up to and including the first eos."""
    is_eos = comp == eos
    first = torch.where(is_eos.any(1), is_eos.float().argmax(1),
                        torch.full_like(is_eos[:, 0], comp.size(1), dtype=torch.long))
    ar = torch.arange(comp.size(1), device=comp.device)[None]
    return (ar <= first[:, None]).long(), first


def policy_update(score, peft_model, c_mask, rewards, group, beta, opt, scaler, autocast,
                  dynamic_sampling: bool = False, token_level: bool = False) -> GRPOStats:
    """Shared GRPO update. score() -> per-token log-probs of the sampled
    completions under the current network; the reference is the same call
    inside peft_model.disable_adapter()."""
    adv = group_advantages(rewards, group)
    rg = rewards.view(-1, group)
    informative = (rg.std(1, unbiased=False) > 1e-6).repeat_interleave(group)
    if dynamic_sampling and not bool(informative.any()):
        # DAPO-style dynamic sampling: groups whose samples all scored the same
        # carry no advantage signal, so the step is skipped instead of letting
        # the KL term alone move the policy
        return GRPOStats(loss=0.0, reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)),
                         zero_var_groups=1.0, kl=0.0, completion_len=float(c_mask.float().sum(1).mean()))
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        with torch.no_grad(), peft_model.disable_adapter():
            ref_lp = score()
        lp = score()
    m = c_mask.float()
    ratio = torch.exp(lp - lp.detach())                     # == 1, keeps the PPO form
    pg = -(ratio * adv[:, None])
    d = ref_lp - lp
    kl = torch.exp(d) - d - 1                               # k3 estimator
    if dynamic_sampling:  # drop uninformative groups from the loss
        m = m * informative.float()[:, None]
    per_tok = (pg + beta * kl) * m
    if token_level:  # DAPO token-level normalisation: long answers are not down-weighted
        loss = per_tok.sum() / m.sum().clamp(min=1)
    else:
        rows = m.sum(1) > 0
        loss = (per_tok.sum(1)[rows] / m.sum(1)[rows]).mean()
    params = [p for g in opt.param_groups for p in g["params"]]
    opt.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return GRPOStats(
        loss=float(loss.item()),
        reward_mean=float(rewards.mean()),
        reward_std=float(rewards.std(unbiased=False)),
        zero_var_groups=float((rg.std(1, unbiased=False) < 1e-6).float().mean()),
        kl=float(((kl.detach() * m).sum() / m.sum().clamp(min=1)).item()),
        completion_len=float(m.sum(1).mean()),
    )


def grpo_step(model, tok, prompts: list[str], golds: list, reward_fn, opt, device, group: int = 4,
              beta: float = 0.04, temperature: float = 1.0, max_new_tokens: int = 96, scaler=None,
              autocast: bool = False) -> GRPOStats:
    """GRPO step for a text LLM (PeftModel)."""
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    p_ids, p_mask = pad_left([encode(tok, p) for p in prompts], pad, device)
    p_ids, p_mask = p_ids.repeat_interleave(group, 0), p_mask.repeat_interleave(group, 0)
    n_prompt = p_ids.size(1)

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        seq = model.generate(input_ids=p_ids, attention_mask=p_mask, do_sample=True, temperature=temperature,
                             max_new_tokens=max_new_tokens, eos_token_id=eos, pad_token_id=pad,
                             **NEUTRAL_DECODING)
    comp = seq[:, n_prompt:]
    c_mask, first = completion_mask(comp, eos)
    texts = [tok.decode(c[: int(f) + 1], skip_special_tokens=True) for c, f in zip(comp, first)]
    gold_rep = [g for g in golds for _ in range(group)]
    rewards = torch.tensor([reward_fn(t, g) for t, g in zip(texts, gold_rep)], device=device, dtype=torch.float32)
    full_mask = torch.cat([p_mask, c_mask], 1)
    model.train()
    return policy_update(lambda: token_logprobs(model, seq, full_mask, n_prompt), model, c_mask, rewards, group,
                         beta, opt, scaler, autocast)


def embeds_logprobs(llm, prompt_emb, prompt_mask, comp, c_mask):
    """Log-probs of completion tokens given a prompt supplied as embeddings."""
    emb = torch.cat([prompt_emb, llm.get_input_embeddings()(comp).to(prompt_emb.dtype)], 1)
    mask = torch.cat([prompt_mask, c_mask], 1)
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    n = prompt_emb.size(1)
    logits = llm(inputs_embeds=emb, attention_mask=mask, position_ids=pos).logits[:, n - 1 : -1].float()
    return torch.gather(F.log_softmax(logits, -1), 2, comp.unsqueeze(-1)).squeeze(-1)


def sllm_grpo_step(sllm, tok, input_features, n_samples, pieces, golds: list, reward_fn, opt, device,
                   group: int = 4, beta: float = 0.04, temperature: float = 1.0, max_new_tokens: int = 96,
                   scaler=None, autocast: bool = False, dynamic_sampling: bool = False,
                   token_level: bool = False) -> GRPOStats:
    """GRPO step for the end-to-end speech LLM (audio -> JSON). The audio
    path (frozen encoder + frozen projector) is computed once per prompt and
    shared by policy and reference; only the LLM's GRPO LoRA is updated."""
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        audio, lens = sllm.audio_embeddings(input_features, n_samples)
        emb, mask, _ = sllm.assemble(audio, lens, pieces, left_pad=True)
    emb, mask = emb.repeat_interleave(group, 0), mask.repeat_interleave(group, 0)
    sllm.llm.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        comp = sllm.llm.generate(inputs_embeds=emb, attention_mask=mask, do_sample=True, temperature=temperature,
                                 max_new_tokens=max_new_tokens, eos_token_id=eos, pad_token_id=pad,
                                 **NEUTRAL_DECODING)
    c_mask, first = completion_mask(comp, eos)
    texts = [tok.decode(c[: int(f) + 1], skip_special_tokens=True) for c, f in zip(comp, first)]
    gold_rep = [g for g in golds for _ in range(group)]
    rewards = torch.tensor([reward_fn(t, g) for t, g in zip(texts, gold_rep)], device=device, dtype=torch.float32)
    sllm.llm.train()
    return policy_update(lambda: embeds_logprobs(sllm.llm, emb, mask, comp, c_mask), sllm.llm, c_mask, rewards,
                         group, beta, opt, scaler, autocast, dynamic_sampling, token_level)
