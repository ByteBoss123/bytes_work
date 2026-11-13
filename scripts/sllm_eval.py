"""Evaluate the speech LLM: WER on LibriSpeech test-clean and SLURP test
(headset / far-field), and SLU with the official SLURP evaluator.

  python scripts/sllm_eval.py --data_root /tmp/data --run_dir runs/sllm --out results/sllm
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from speechlab import slurp as S  # noqa: E402
from speechlab import speechdata as D  # noqa: E402
from speechlab import speechllm as SL  # noqa: E402
from speechlab.metrics import wer, whisper_normalizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--run_dir", required=True)
ap.add_argument("--out", default="results/sllm")
ap.add_argument("--whisper", default="openai/whisper-small")
ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max_per_set", type=int, default=0, help="0 = full sets")
ap.add_argument("--num_workers", type=int, default=4)
ap.add_argument("--device", default="cuda")
ap.add_argument("--grpo_dir", default=None, help="optional sllm_grpo.py output applied on top of the merged SFT model")
args = ap.parse_args()

root, out = Path(args.data_root), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
device = args.device if torch.cuda.is_available() else "cpu"
model, tok = SL.build(args.whisper, args.llm, args.lora_r, device)
from peft import set_peft_model_state_dict  # noqa: E402
from peft.utils import load_peft_weights  # noqa: E402

set_peft_model_state_dict(model.llm, load_peft_weights(str(Path(args.run_dir) / "lora"), device=device))
model.projector.load_state_dict(torch.load(Path(args.run_dir) / "projector.pt", map_location=device))
if args.grpo_dir:
    from peft import PeftModel  # noqa: E402

    model.llm = PeftModel.from_pretrained(model.llm.merge_and_unload(), str(Path(args.grpo_dir) / "lora"))
model.eval()
fe = SL.feature_extractor(args.whisper)
norm = whisper_normalizer(args.whisper)
enc_dtype = next(model.encoder.parameters()).dtype


def run(items):
    ds = D.ItemDataset(items)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers,
                                     collate_fn=D.Collator(fe, tok, items, with_target=False))
    preds = [None] * len(items)
    for idx, feats, n, pieces in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            texts = model.generate(feats.to(device, dtype=enc_dtype), n, pieces, tok)
        for i, t in zip(idx, texts):
            preds[i] = t
    return preds


def cap(items):
    return items[: args.max_per_set] if args.max_per_set else items


slurp_dir = root / "slurp_repo" / "dataset" / "slurp"
results = {"run_dir": args.run_dir, "grpo_dir": args.grpo_dir, "normalizer": "whisper EnglishTextNormalizer", "sets": {}}
if (root / "LibriSpeech" / "test-clean").exists():
    its, _ = D.filter_by_duration(cap(D.librispeech_items(root / "LibriSpeech", "test-clean")))
    p = run(its)
    results["sets"]["librispeech_test_clean"] = wer([i["text"] for i in its], p, norm)
    with open(out / "pred_librispeech_test_clean.jsonl", "w") as f:
        for i, t in zip(its, p):
            f.write(json.dumps({"id": i["id"], "ref": i["text"], "hyp": t}) + "\n")

for cond, far in (("headset", False), ("farfield", True)):
    its = D.slurp_items(slurp_dir, root / "slurp_audio" / "slurp_real", "test", tasks=("asr", "slu"),
                        farfield=(far,))
    asr = cap([i for i in its if i["task"] == "asr"])
    keep = {i["id"] for i in asr}
    slu = [i for i in its if i["task"] == "slu" and i["id"] in keep]
    p_asr = run(asr)
    results["sets"][f"slurp_test_{cond}_asr"] = wer([i["text"] for i in asr], p_asr, norm)
    p_slu = run(slu)
    parsed = [S.parse_prediction(t) for t in p_slu]
    pred_file = out / f"pred_slurp_test_{cond}_slu.jsonl"
    with open(pred_file, "w") as f:
        for i, pr in zip(slu, parsed):
            f.write(json.dumps(S.prediction_record(i["id"], pr)) + "\n")
    off = out / f"official_slurp_{cond}.json"
    subprocess.run([sys.executable, str(HERE / "slurp_official_eval.py"), str(root / "slurp_repo"),
                    str(slurp_dir / "test.jsonl"), str(pred_file), str(off)], check=True, capture_output=True)
    results["sets"][f"slurp_test_{cond}_slu"] = {
        "n_recordings": len(slu),
        "json_parse_rate": round(sum(x is not None for x in parsed) / max(1, len(parsed)), 4),
        "official": json.load(open(off)),
    }
    print(cond, json.dumps(results["sets"][f"slurp_test_{cond}_slu"]["official"].get("slu f1")))

json.dump(results, open(out / "sllm_eval.json", "w"), indent=2)
print(json.dumps({k: (v.get("wer") if "wer" in v else v["official"].get("intent (scen_act)")) for k, v in results["sets"].items()}, indent=2))
