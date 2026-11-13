"""End-to-end smoke tests of the speech-LLM and SLU/GRPO scripts with tiny
random models on the fixture data (mechanics only, no reported numbers)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from fixtures import make_slurp  # noqa: E402
from speechlab import slurp as S  # noqa: E402

OFFICIAL = ROOT.parent / "ext" / "slurp"


def _run(args):
    env = dict(os.environ, SPEECHLAB_TEST_TINY="1")
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env=env, timeout=1200)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-5000:]
    return out


@pytest.fixture()
def data_root(tmp_path):
    if not OFFICIAL.exists():
        pytest.skip("official SLURP repo not cloned")
    root = tmp_path / "data"
    ann, audio = make_slurp(tmp_path / "raw", n_speakers=4, utts_per_speaker=8)
    (root / "slurp_repo" / "dataset").mkdir(parents=True)
    shutil.copytree(ann, root / "slurp_repo" / "dataset" / "slurp")
    shutil.copytree(OFFICIAL / "scripts", root / "slurp_repo" / "scripts")
    (root / "slurp_audio").mkdir()
    shutil.copytree(audio, root / "slurp_audio" / "slurp_real")
    return root


def test_speech_llm_ddp_train_and_eval(data_root, tmp_path):
    run = tmp_path / "sllm"
    _run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "scripts/sllm_train.py",
          "--data_root", str(data_root), "--out_dir", str(run), "--max_steps", "4", "--batch", "2",
          "--grad_accum", "2", "--num_workers", "0", "--log_every", "2", "--grad_ckpt", "0"])
    st = json.load(open(run / "train_stats.json"))
    assert st["world_size"] == 2 and st["steps"] == 4
    out = tmp_path / "res"
    _run([sys.executable, "scripts/sllm_eval.py", "--data_root", str(data_root), "--run_dir", str(run),
          "--out", str(out), "--batch", "4", "--num_workers", "0", "--device", "cpu"])
    res = json.load(open(out / "sllm_eval.json"))
    assert {"slurp_test_headset_asr", "slurp_test_farfield_slu"} <= set(res["sets"])

    grpo = tmp_path / "sllm_grpo"
    _run([sys.executable, "scripts/sllm_grpo.py", "--data_root", str(data_root), "--sft_dir", str(run),
          "--out_dir", str(grpo), "--steps", "3", "--prompts", "2", "--group", "3", "--eval_every", "2",
          "--eval_n", "4", "--eval_batch", "2"])
    log = json.load(open(grpo / "grpo_stats.json"))["log"]
    assert log[0]["step"] == 0 and "dev_reward" in log[0] and log[-1]["step"] == 3
    out2 = tmp_path / "res_grpo"
    _run([sys.executable, "scripts/sllm_eval.py", "--data_root", str(data_root), "--run_dir", str(run),
          "--grpo_dir", str(grpo), "--out", str(out2), "--batch", "4", "--num_workers", "0", "--device", "cpu"])
    assert json.load(open(out2 / "sllm_eval.json"))["grpo_dir"] == str(grpo)


def test_slu_sft_grpo_eval(data_root, tmp_path):
    asr = tmp_path / "asr"
    asr.mkdir()
    slurp_dir = data_root / "slurp_repo" / "dataset" / "slurp"
    for split in ("train", "test", "devel"):
        with open(asr / f"hyp_slurp_{split}.jsonl", "w") as f:
            for ex in S.load_jsonl(slurp_dir / f"{split}.jsonl"):
                for r in ex["recordings"]:
                    f.write(json.dumps({"id": r["file"], "slurp_id": ex["slurp_id"],
                                        "farfield": not r["file"].endswith("-headset.flac"),
                                        "ref": S.transcript(ex), "hyp": S.transcript(ex)}) + "\n")
    common = ["--data_root", str(data_root), "--asr_dir", str(asr), "--run_dir", str(tmp_path / "slu")]
    _run([sys.executable, "scripts/slu_text.py", "sft", *common, "--sft_batch", "4", "--sft_epochs", "1"])
    _run([sys.executable, "scripts/slu_text.py", "grpo", *common, "--grpo_steps", "3", "--grpo_prompts", "2",
          "--group", "3", "--eval_every", "2", "--eval_n", "4"])
    log = json.load(open(tmp_path / "slu" / "grpo_stats.json"))["log"]
    assert log[0]["step"] == 0 and "dev_reward" in log[0] and log[-1]["step"] == 3
    _run([sys.executable, "scripts/slu_text.py", "eval", *common, "--out", str(tmp_path / "out")])
    res = json.load(open(tmp_path / "out" / "slu_eval.json"))
    assert {"sft|asr_headset", "sft+grpo|gold_transcript"} <= set(res)


def test_grpo_increases_reward_on_toy_task():
    """Sign/plumbing check: GRPO must raise a simple reward (completions that
    contain the letter 'a') on a tiny random policy."""
    import torch
    from peft import LoraConfig, get_peft_model

    os.environ["SPEECHLAB_TEST_TINY"] = "1"
    from speechlab import grpo as G
    from speechlab.speechllm import tiny_llm

    llm, tok = tiny_llm()
    model = get_peft_model(llm, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    for n, p in model.named_parameters():  # let the policy move fast on a toy problem
        if "lm_head" in n:
            p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    reward = lambda text, _g: text.count("a") / max(1, len(text))  # noqa: E731
    prompts = [G.slu_prompt_text("set alarm")] * 4
    hist = []
    torch.manual_seed(0)
    for _ in range(60):
        st = G.grpo_step(model, tok, prompts, [None] * 4, reward, opt, "cpu", group=8, beta=0.0,
                         max_new_tokens=12)
        hist.append(st.reward_mean)
    assert sum(hist[-10:]) / 10 > sum(hist[:10]) / 10 + 0.05, hist


def test_decoding_ignores_shipped_generation_config():
    """Qwen ships repetition_penalty/top_k/top_p defaults; our generate calls
    must neutralise them so greedy output == argmax of the scored log-probs."""
    import torch

    os.environ["SPEECHLAB_TEST_TINY"] = "1"
    from speechlab import grpo as G
    from speechlab.speechllm import tiny_llm

    llm, tok = tiny_llm()
    llm.generation_config.repetition_penalty = 3.0
    llm.generation_config.top_k = 2
    llm.eval()
    ids, mask = G.pad_left([G.encode(tok, G.slu_prompt_text("set alarm"))], tok.pad_token_id, "cpu")
    out = llm.generate(input_ids=ids, attention_mask=mask, max_new_tokens=10, do_sample=False,
                       eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id,
                       output_scores=True, return_dict_in_generate=True, **G.NEUTRAL_DECODING)
    seq = out.sequences
    n = ids.size(1)
    logits = llm(input_ids=seq, attention_mask=torch.ones_like(seq)).logits[0, n - 1 : -1]
    assert torch.equal(logits.argmax(-1), seq[0, n:])


def test_speech_llm_scoring_matches_generation_and_grpo_runs():
    """Embeds-based log-probs must equal generate() scores (left-padded batch
    of different audio lengths), and one speech-LLM GRPO step must update only
    the GRPO adapter, leaving encoder and projector untouched."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model

    os.environ["SPEECHLAB_TEST_TINY"] = "1"
    from speechlab import grpo as G
    from speechlab import speechllm as SL

    m, tok = SL.build(device="cpu")
    m.llm = m.llm.merge_and_unload()  # "SFT" merged into the base, as in sllm_grpo.py
    m.llm = get_peft_model(m.llm, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    for p in m.projector.parameters():
        p.requires_grad_(False)
    fe = SL.feature_extractor()
    rng = np.random.default_rng(0)
    wavs = [rng.standard_normal(16000).astype("float32") * 0.1, rng.standard_normal(56000).astype("float32") * 0.1]
    feats = fe(wavs, sampling_rate=16000, return_tensors="pt").input_features
    n = [len(w) for w in wavs]
    pieces = [SL.prompt_pieces(tok, "slu", None)] * 2
    m.eval()
    with torch.no_grad():
        audio, lens = m.audio_embeddings(feats, n)
        emb, mask, _ = m.assemble(audio, lens, pieces, left_pad=True)
        out = m.llm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=6, do_sample=False,
                             pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True,
                             **G.NEUTRAL_DECODING)
        comp = out.sequences
        assert comp.size(1) <= 6  # only new tokens are returned for inputs_embeds
        c_mask = torch.ones_like(comp)
        lp = G.embeds_logprobs(m.llm, emb, mask, comp, c_mask)
        sc = torch.stack(out.scores, 1).log_softmax(-1).gather(2, comp.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(lp, sc, atol=1e-5)

    before = {k: v.clone() for k, v in m.named_parameters()}
    opt = torch.optim.Adam([p for p in m.llm.parameters() if p.requires_grad], lr=1e-2)
    gold = {"scenario": "calendar", "action": "set", "entities": []}
    st = G.sllm_grpo_step(m, tok, feats, n, pieces, [gold, gold], lambda t, g: float(len(t) % 3), opt, "cpu",
                          group=3, max_new_tokens=6)
    assert np.isfinite(st.loss)
    changed = {k for k, v in m.named_parameters() if not torch.equal(v, before[k])}
    assert changed and all("lora_" in k for k in changed), changed
