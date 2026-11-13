# Speaker-Aware Speech LLM

Speaker verification on far-field voice-assistant audio, a speech LLM that maps audio directly to transcripts and intent JSON, and reinforcement learning (GRPO) with a verifiable reward for spoken-language understanding. Everything runs on real public speech data and fits on Kaggle's 2x T4 GPUs.

| Component | What it does | Data |
|---|---|---|
| **1. Speaker verification** | Pretrained ECAPA-TDNN, cosine and AS-norm scoring, headset vs far-field mic conditions, babble-noise robustness, DDP fine-tuning on held-out-speaker splits | SLURP real recordings (72,395 recordings, 174 speakers), VoxCeleb1-O |
| **2. Speech LLM** | Frozen Whisper-small encoder, frame-stacking MLP projector, Qwen2.5-0.5B-Instruct with LoRA, multitask ASR + SLU, DDP training | LibriSpeech train-clean-100 / test-clean, SLURP |
| **3. RL for SLU** | SFT then GRPO (own implementation) with a verifiable reward: JSON format + exact intent + entity F1, k3 KL to the SFT reference. Applied twice: to the end-to-end speech LLM (audio → JSON) and to a text cascade on Whisper transcripts | SLURP audio; Whisper-small transcripts of it |
| **4. Device-directed speech detection** | Whisper transcript (text) + Whisper encoder (acoustic) classifiers and late fusion. Leave-corpus-out folds score every model against a channel-only and a length-only shortcut probe | SLURP and MInDS-14 (directed), AMI meetings and VoxCeleb1 (non-directed) |
| **5. Scale** | Same speech LLM at 0.5B / 1.5B / 3B LLM size (same data, steps and global batch), with throughput and peak memory. A runbook repeats the 1→N GPU scaling on 4–8 GPU AWS instances | as above |
| **Evaluation** | VoxCeleb1-O reproduction check, ~2.0M SLURP verification trials, the official SLURP SLU metric (unmodified evaluator), Whisper-normalised WER | as above |

## Design decisions worth knowing

* **Speaker-disjoint split.** The official SLURP train/devel/test splits share speakers (136 of 142 test speakers appear in train). Speaker verification therefore uses its own split: 40 held-out speakers (25,012 recordings) and 134 training speakers (47,384 recordings), chosen by seed among speakers with at least 30 headset and 30 far-field recordings.
* **Same-take leakage guard.** SLURP's headset and far-field files of one take are simultaneous recordings of the same utterance (shared `recid`). Any test recording whose take is in the enrollment set is removed from that model's trials (about 110 per condition).
* **Pipeline check before new claims.** `sv_vox.py` re-scores the pretrained model on VoxCeleb1-O (cleaned, 37,611 trials) against the model card's 0.80% EER, so a broken pipeline shows up before any SLURP number is trusted.
* **Honest scaling numbers.** Fine-tuning reads audio from a pre-decoded int16 memmap, because Kaggle has 4 vCPUs for 2 GPUs and FLAC decoding would otherwise be the bottleneck. Each run logs the fraction of time spent waiting on data next to its samples/s.
* **Operating points, not only EER.** Each condition reports the false-reject rate at 1% and 0.1% false-accept. That is the trade-off an assistant makes when it decides whether the enrolled user is the one speaking.
* **Fair WER.** Whisper's EnglishTextNormalizer is applied to references and hypotheses of every system. SLURP spells numbers out ("seven am") and Whisper writes digits, so unnormalised WER would penalise the baseline unfairly.
* **Official SLU metric, unmodified.** `scripts/slurp_official_eval.py` runs the evaluator from the SLURP repo as is. The only change is a keyword shim for the jiwer API rename (`truth=` became `reference=`). A unit test feeds gold back in and checks that SLU-F1 = 1.0.
* **GRPO correctness checks.** Log-probs used for the policy gradient match `generate()` scores exactly under left padding (tested). A toy-reward test shows the reward rising, which checks the sign and plumbing of the update.

## Run (Kaggle, GPU T4 x2, Internet ON)

| Notebook | Main outputs |
|---|---|
| `notebooks/01_speaker_verification.ipynb` | `results/sv_vox1o*.json`, `results/sv/*.json`, `runs/ecapa_ft/train_stats_ws{1,2}.json` |
| `notebooks/02_speech_llm.ipynb` | `results/asr/wer_*.json`, `results/sllm/sllm_eval.json`, `results/sllm_grpo/sllm_eval.json` |
| `notebooks/03_slu_grpo.ipynb` | `runs/slu/grpo_stats.json`, `results/slu/slu_eval.json` |
| `notebooks/04_directed_speech.ipynb` | `results/ddsd/ddsd_eval.json` |
| `notebooks/05_llm_scaling.ipynb` | `runs/sllm_scale_*/train_stats.json`, `results/sllm_scale_*/sllm_eval.json` |

Each notebook ends with `scripts/summarize.py`, which writes `results/SUMMARY.md` by copying numbers verbatim from the result JSONs.

Data comes from the original sources via `scripts/prepare_data.sh`:
* SLURP audio from Zenodo 4274930 (CC BY-NC 4.0)
* SLURP annotations from github.com/pswietojanski/slurp
* VoxCeleb1 test audio from the Hugging Face mirror ProgramComputer/voxceleb; trial list `veri_test2.txt` from the VGG site
* LibriSpeech from OpenSLR 12

## Tests

`pytest tests` (CPU, about 1.5 minutes; 26 tests) covers:
* metrics against scikit-learn
* a planted recording-channel shortcut: the channel-only probe scores near-perfect in-domain and collapses across corpora, while the real signal transfers
* directed-speech data prep from Hugging Face-style parquet audio, including 8 kHz to 16 kHz resampling
* `prepare_data.sh` against a local server with nested archive layouts, a dead mirror and missing trial files
* duration-capped batching and the int16 audio cache
* SNR mixing accuracy
* the split and leakage guard
* the official-evaluator round trip
* 2-process DDP runs of the speaker fine-tune and the speech-LLM trainer (gloo)
* the SFT → GRPO → eval pipeline
* GRPO log-prob equality for text and audio-embedding prompts, a check that speech-LLM GRPO updates only its LoRA, the toy-reward check, and decoding that ignores Qwen's shipped sampling defaults

The tests use random tiny models and synthetic tones only to exercise code paths. **No reported number comes from them.**

## Limits (read before quoting results)

* **Directed-speech detection measures the corpus confound rather than removing it.** No public dataset records directed and non-directed speech under matched conditions. Leave-corpus-out folds and the shortcut probes show how much of each model's score survives a change of corpus; only that part counts. The negatives (meetings, interviews) contain no command-like speech addressed to people, so this is an easier task than production false-trigger mitigation. MInDS-14 publishes no speaker IDs, so its in-domain split is by recording.
* **Scale is bounded by the hardware.** On 2x T4 the largest LLM is 3B (LoRA). `docs/SCALE_OUT_AWS.md` repeats the GPU-scaling runs on 4–8 GPU instances, which cost money in your AWS account. This is not foundation-model pretraining.
* **Qwen2.5-3B is under the Qwen research licence (non-commercial).** The 0.5B and 1.5B models are Apache 2.0.
* **Speech-LLM GRPO keeps the audio path frozen.** Encoder and projector are fixed, and only a new LoRA on the LLM is trained, so the reference model sees the same audio tokens.
* **Possible pretraining overlap.** Whisper's and Qwen's pretraining data may overlap public benchmarks such as LibriSpeech. The pretrained ECAPA model was trained on VoxCeleb1 dev + VoxCeleb2 dev; VoxCeleb1-O is its held-out test set.
* **Different split protocols.** Speech-LLM and SLU numbers use the official SLURP test split, so they are comparable to published SLURP results. Speaker-verification numbers use the speaker-disjoint split above.
