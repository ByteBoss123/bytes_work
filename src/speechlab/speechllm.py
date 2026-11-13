"""Speech LLM: frozen Whisper encoder -> frame-stacking MLP projector ->
decoder-only LLM (Qwen2.5) adapted with LoRA.

The audio tokens are spliced into a chat prompt between <audio> and
</audio>; only the assistant answer is supervised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

SYSTEM = "You are a helpful voice assistant."
TASKS = {
    "asr": "Transcribe the audio exactly, in lower case.",
    "slu": "Parse the voice-assistant request in the audio into JSON with scenario, action and entities.",
    "intent": "Name the intent of the spoken voice-assistant request. Answer with the intent label only.",
}
SAMPLES_PER_ENC_FRAME = 320  # whisper: hop 160 samples, conv stride 2
MAX_ENC_FRAMES = 1500        # 30 s


def enc_frames(n_samples: int) -> int:
    return min(MAX_ENC_FRAMES, math.ceil(n_samples / SAMPLES_PER_ENC_FRAME))


class Projector(nn.Module):
    def __init__(self, enc_dim: int, llm_dim: int, stack: int = 4):
        super().__init__()
        self.stack = stack
        self.net = nn.Sequential(nn.Linear(enc_dim * stack, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        t2 = t // self.stack
        h = h[:, : t2 * self.stack].reshape(b, t2, d * self.stack)
        return self.net(h)


@dataclass
class PromptPieces:
    prefix: list[int]
    suffix: list[int]
    target: list[int] | None


def prompt_pieces(tokenizer, task: str, target_text: str | None) -> PromptPieces:
    pre = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<audio>"
    suf = f"</audio>\n{TASKS[task]}<|im_end|>\n<|im_start|>assistant\n"
    enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    tgt = enc(target_text + "<|im_end|>") if target_text is not None else None
    return PromptPieces(enc(pre), enc(suf), tgt)


class SpeechLLM(nn.Module):
    def __init__(self, encoder: nn.Module, llm: nn.Module, enc_dim: int, llm_dim: int, stack: int = 4):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.projector = Projector(enc_dim, llm_dim, stack)
        self.llm = llm
        self.stack = stack

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # frozen encoder always in eval mode
        return self

    def audio_embeddings(self, input_features: torch.Tensor, n_samples: list[int]):
        with torch.no_grad():
            h = self.encoder(input_features).last_hidden_state
        a = self.projector(h.to(self.projector.net[0].weight.dtype))
        lens = [max(1, enc_frames(n) // self.stack) for n in n_samples]
        return a, lens

    def _embed_ids(self, ids: list[int], device) -> torch.Tensor:
        return self.llm.get_input_embeddings()(torch.tensor(ids, device=device))

    def assemble(self, audio: torch.Tensor, lens: list[int], pieces: list[PromptPieces], left_pad: bool):
        """Return inputs_embeds, attention_mask, labels (labels None when no target)."""
        device = audio.device
        seqs, labs = [], []
        for i, pc in enumerate(pieces):
            parts = [self._embed_ids(pc.prefix, device), audio[i, : lens[i]].to(self.llm.dtype),
                     self._embed_ids(pc.suffix, device)]
            n_ctx = len(pc.prefix) + lens[i] + len(pc.suffix)
            lab = [-100] * n_ctx
            if pc.target is not None:
                parts.append(self._embed_ids(pc.target, device))
                lab += pc.target
            seqs.append(torch.cat(parts, 0))
            labs.append(lab)
        L = max(s.size(0) for s in seqs)
        H = seqs[0].size(1)
        emb = torch.zeros(len(seqs), L, H, device=device, dtype=seqs[0].dtype)
        mask = torch.zeros(len(seqs), L, device=device, dtype=torch.long)
        labels = torch.full((len(seqs), L), -100, device=device, dtype=torch.long)
        for i, s in enumerate(seqs):
            n = s.size(0)
            sl = slice(L - n, L) if left_pad else slice(0, n)
            emb[i, sl] = s
            mask[i, sl] = 1
            labels[i, sl] = torch.tensor(labs[i], device=device)
        has_target = any(pc.target is not None for pc in pieces)
        return emb, mask, (labels if has_target else None)

    def forward(self, input_features, n_samples, pieces):
        audio, lens = self.audio_embeddings(input_features, n_samples)
        emb, mask, labels = self.assemble(audio, lens, pieces, left_pad=False)
        return self.llm(inputs_embeds=emb, attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def generate(self, input_features, n_samples, pieces, tokenizer, max_new_tokens=96):
        audio, lens = self.audio_embeddings(input_features, n_samples)
        emb, mask, _ = self.assemble(audio, lens, pieces, left_pad=True)
        eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        out = self.llm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new_tokens,
                                do_sample=False, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id or eos,
                                repetition_penalty=1.0)  # override Qwen's shipped 1.05
        return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]


def build(whisper_name="openai/whisper-small", llm_name="Qwen/Qwen2.5-0.5B-Instruct", lora_r=16,
          device="cuda", dtype=torch.float16, stack=4):
    if _tiny():
        return build_tiny(device if torch.cuda.is_available() else "cpu")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperModel

    enc = WhisperModel.from_pretrained(whisper_name, dtype=dtype).encoder
    llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=dtype, attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(llm_name)
    llm = get_peft_model(llm, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                         "gate_proj", "up_proj", "down_proj"]))
    model = SpeechLLM(enc, llm, enc.config.d_model, llm.config.hidden_size, stack)
    return model.to(device), tok


def feature_extractor(whisper_name="openai/whisper-small"):
    from transformers import WhisperFeatureExtractor

    if _tiny():
        return WhisperFeatureExtractor(feature_size=80)
    return WhisperFeatureExtractor.from_pretrained(whisper_name)


def _tiny() -> bool:
    import os

    return os.environ.get("SPEECHLAB_TEST_TINY") == "1"


def tiny_tokenizer():
    """Unit tests only: a small byte-level BPE with the Qwen chat specials."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    corpus = [SYSTEM, *TASKS.values(), "<audio></audio> user assistant system",
              '{"scenario":"calendar","action":"set","entities":[{"type":"time","filler":"seven"}]}',
              "set alarm for seven what is the weather"]
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=specials,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=specials[1:])
    return tok


def tiny_llm():
    """Unit tests only: randomly initialised 2-layer Qwen2 + tiny tokenizer."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    tok = tiny_tokenizer()
    qc = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                     eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id)
    return Qwen2ForCausalLM(qc), tok


def build_tiny(device="cpu", lora_r=4):
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    llm, tok = tiny_llm()
    torch.manual_seed(1)
    wc = WhisperConfig(d_model=32, encoder_layers=1, encoder_attention_heads=2, decoder_layers=1,
                       decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=80,
                       max_source_positions=1500)
    enc = WhisperEncoder(wc)
    llm = get_peft_model(llm, LoraConfig(r=lora_r, lora_alpha=2 * lora_r,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    return SpeechLLM(enc, llm, 32, 32).to(device), tok
