from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import torch


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


CONFIGURATION_PY = r'''
from transformers import PretrainedConfig


class NanoChatConfig(PretrainedConfig):
    model_type = "nanochat"

    def __init__(
        self,
        sequence_len=1024,
        vocab_size=32768,
        n_layer=22,
        n_head=22,
        n_kv_head=22,
        n_embd=1408,
        window_pattern="L",
        bos_token_id=32759,
        eos_token_id=32759,
        pad_token_id=32759,
        representation_layer=-1,
        representation_hidden_states_mode="all",
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern
        self.hidden_size = n_embd
        self.max_position_embeddings = sequence_len
        self.is_encoder_decoder = False
        self.representation_layer = representation_layer
        self.representation_hidden_states_mode = representation_hidden_states_mode
'''


MODELING_PY = r'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from .configuration_nanochat import NanoChatConfig


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class NanoChatLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = NanoChatLinear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = NanoChatLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = NanoChatLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = NanoChatLinear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = NanoChatLinear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q).to(v.dtype) * 1.2, norm(k).to(v.dtype) * 1.2
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        if window_size[0] >= T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = torch.ones((T, T), dtype=torch.bool, device=x.device).tril()
            left = window_size[0]
            pos = torch.arange(T, device=x.device)
            mask &= (pos[None, :] >= (pos[:, None] - left))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = NanoChatLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = NanoChatLinear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class NanoChatPreTrainedModel(PreTrainedModel):
    config_class = NanoChatConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = False
    all_tied_weights_keys = {}

    def _init_weights(self, module):
        return


class NanoChatModel(NanoChatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = NanoChatLinear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = NanoChatLinear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(config.vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len * 10, head_dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _hidden_states(self, input_ids, collect_layers=False):
        B, T = input_ids.size()
        cos_sin = self.cos[:, :T].to(device=input_ids.device), self.sin[:, :T].to(device=input_ids.device)
        compute_dtype = self.transformer.wte.weight.dtype
        x = norm(self.transformer.wte(input_ids).to(compute_dtype))
        if T == 1:
            filler = torch.full_like(input_ids, self.config.bos_token_id)
            result = self._hidden_states(torch.cat([input_ids, filler], dim=1), collect_layers=collect_layers)
            if collect_layers:
                final_hidden, layer_hiddens = result
                return final_hidden[:, :1, :], tuple(h[:, :1, :] for h in layer_hiddens)
            return result[:, :1, :]
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        x0 = x
        x_backout = None
        backout_layer = self.config.n_layer // 2
        layer_hiddens = []
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i].to(x.dtype) * x + self.x0_lambdas[i].to(x.dtype) * x0
            ve = self.value_embeds[str(i)](input_ids).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            if i == backout_layer:
                x_backout = x
            if collect_layers:
                layer_hiddens.append(norm(x))
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        final_hidden = norm(x)
        if collect_layers:
            layer_hiddens.append(final_hidden)
            return final_hidden, tuple(layer_hiddens)
        return final_hidden

    def _select_representation(self, final_hidden, layer_hiddens):
        layer = int(getattr(self.config, "representation_layer", -1))
        if layer < 0:
            layer = len(layer_hiddens) + layer
        layer = max(0, min(layer, len(layer_hiddens) - 1))
        return layer_hiddens[layer]

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, return_dict=True, **kwargs):
        del attention_mask, kwargs
        final_hidden, layer_hiddens = self._hidden_states(input_ids, collect_layers=True)
        hidden = self._select_representation(final_hidden, layer_hiddens).float()
        hidden_states = None
        if output_hidden_states:
            mode = getattr(self.config, "representation_hidden_states_mode", "all")
            if mode == "selected_only":
                hidden_states = (hidden,)
            elif mode == "selected_last":
                hidden_states = tuple(h.float() for h in layer_hiddens) + (hidden,)
            else:
                hidden_states = tuple(h.float() for h in layer_hiddens)
        if not return_dict:
            return (hidden,)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=hidden_states)


class NanoChatForCausalLM(NanoChatModel):
    _tied_weights_keys = []

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=True, **kwargs):
        del attention_mask, kwargs
        hidden = self._hidden_states(input_ids)
        logits = self.lm_head(hidden)[..., : self.config.vocab_size].float()
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        if not return_dict:
            out = (logits,)
            return ((loss,) + out) if loss is not None else out
        return CausalLMOutput(loss=loss, logits=logits)
'''


TOKENIZATION_PY = r'''
import os
import pickle

import torch
from transformers import BatchEncoding, PreTrainedTokenizer


class NanoChatTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"tokenizer_file": "tokenizer.pkl"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer_file=None, **kwargs):
        if tokenizer_file is None:
            tokenizer_file = os.path.join(kwargs.get("name_or_path", "."), "tokenizer.pkl")
        with open(tokenizer_file, "rb") as f:
            self.enc = pickle.load(f)
        bos = "<|bos|>"
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("pad_token", None)
        kwargs.pop("unk_token", None)
        super().__init__(
            bos_token=bos,
            eos_token=bos,
            pad_token=bos,
            unk_token=None,
            model_max_length=kwargs.pop("model_max_length", 1024),
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def get_vocab(self):
        vocab = {self.enc.decode([i]): i for i in range(self.enc.n_vocab)}
        return vocab

    def _tokenize(self, text, **kwargs):
        return [self.enc.decode([i]) for i in self.enc.encode_ordinary(text)]

    def _convert_token_to_id(self, token):
        try:
            return self.enc.encode_single_token(token)
        except Exception:
            ids = self.enc.encode_ordinary(token)
            return ids[0] if ids else self.bos_token_id

    def _convert_id_to_token(self, index):
        return self.enc.decode([index])

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            ids = [ids]
        return [self.enc.decode([i]) for i in ids if not (skip_special_tokens and i == self.bos_token_id)]

    def encode(self, text, add_special_tokens=True, **kwargs):
        ids = self.enc.encode_ordinary(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return ids

    def _encode_with_offsets(self, text, add_special_tokens=True):
        ids = self.enc.encode_ordinary(text)
        decoded, starts = self.enc.decode_with_offsets(ids)
        del decoded
        offsets = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            offsets.append((start, end))
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
            offsets = [(0, 0)] + offsets
        return ids, offsets

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i != self.bos_token_id]
        return self.enc.decode(token_ids)

    def __call__(self, text=None, text_pair=None, return_tensors=None, padding=False, truncation=False, max_length=None, add_special_tokens=True, is_split_into_words=False, return_special_tokens_mask=False, **kwargs):
        if text is None:
            text = kwargs.get("text", "")
        if isinstance(text, tuple):
            items = [text]
        elif isinstance(text, list) and is_split_into_words and all(isinstance(x, str) for x in text):
            items = [text]
        elif isinstance(text, list):
            items = text
        else:
            items = [text]
        encoded, masks, special_masks, offset_mappings = [], [], [], []
        want_offsets = kwargs.get("return_offsets_mapping", False)
        for item in items:
            if is_split_into_words:
                parts = item
                ids = [self.bos_token_id] if add_special_tokens else []
                sm = [1] if add_special_tokens else []
                offsets = [(0, 0)] if add_special_tokens else []
                cursor = 0
                for part in parts:
                    part_text = str(part)
                    part_ids = self.enc.encode_ordinary(part_text)
                    ids.extend(part_ids)
                    sm.extend([0] * len(part_ids))
                    _, part_starts = self.enc.decode_with_offsets(part_ids)
                    for j, start in enumerate(part_starts):
                        end = part_starts[j + 1] if j + 1 < len(part_starts) else len(part_text)
                        offsets.append((cursor + start, cursor + end))
                    cursor += len(part_text)
            else:
                if text_pair is not None:
                    item = str(item) + "\n" + str(text_pair)
                elif isinstance(item, tuple):
                    item = "\n".join(str(x) for x in item)
                ids, offsets = self._encode_with_offsets(str(item), add_special_tokens=add_special_tokens)
                sm = [0] * len(ids)
                if add_special_tokens and sm:
                    sm[0] = 1
            if truncation and max_length is not None:
                ids = ids[:max_length]
                sm = sm[:max_length]
                offsets = offsets[:max_length]
            encoded.append(ids)
            masks.append([1] * len(ids))
            special_masks.append(sm)
            offset_mappings.append(offsets)
        if padding:
            pad_to = max(len(x) for x in encoded) if encoded else 0
            if padding == "max_length" and max_length is not None:
                pad_to = max_length
            for ids, mask, sm, offsets in zip(encoded, masks, special_masks, offset_mappings):
                pad_len = max(0, pad_to - len(ids))
                ids.extend([self.pad_token_id] * pad_len)
                mask.extend([0] * pad_len)
                sm.extend([1] * pad_len)
                offsets.extend([(0, 0)] * pad_len)
        data = {"input_ids": encoded, "attention_mask": masks}
        if return_special_tokens_mask:
            data["special_tokens_mask"] = special_masks
        if want_offsets:
            data["offset_mapping"] = offset_mappings
        if return_tensors == "pt":
            data = {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}
        elif len(items) == 1:
            data = {k: v[0] for k, v in data.items()}
        return BatchEncoding(data)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        name = (filename_prefix + "-" if filename_prefix else "") + "tokenizer.pkl"
        path = os.path.join(save_directory, name)
        with open(path, "wb") as f:
            pickle.dump(self.enc, f)
        return (path,)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_PROJECT_DIR / "chinese_cache/base_checkpoints/zh-d22-2gpu"),
    )
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_PROJECT_DIR / "chinese_cache/tokenizer"))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PROJECT_DIR / "chinese_cache/hf_models/zh-d22-step10000"),
    )
    parser.add_argument(
        "--representation-layer",
        type=int,
        default=-1,
        help=(
            "Layer exposed as AutoModel.last_hidden_state for representation tasks. "
            "0 is after block 0; n_layer-1 is after the last block before backout; -1 is final normalized output."
        ),
    )
    parser.add_argument(
        "--representation-hidden-states-mode",
        choices=["all", "selected_only", "selected_last"],
        default="all",
        help=(
            "Controls AutoModel output_hidden_states for representation tasks. "
            "'all' exposes every layer; 'selected_only' makes hidden_states[-1] equal "
            "the selected representation layer for official CogBench layer sweeps; "
            "'selected_last' appends the selected representation after all layers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / f"model_{args.step:06d}.pt"
    meta_path = checkpoint_dir / f"meta_{args.step:06d}.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    model_config = dict(meta["model_config"])
    model_config.setdefault("window_pattern", "L")

    tokenizer_pkl = Path(args.tokenizer_dir) / "tokenizer.pkl"
    with tokenizer_pkl.open("rb") as f:
        enc = pickle.load(f)
    bos_token_id = enc.encode_single_token("<|bos|>")

    config = {
        "model_type": "nanochat",
        "architectures": ["NanoChatForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_nanochat.NanoChatConfig",
            "AutoModel": "modeling_nanochat.NanoChatModel",
            "AutoModelForCausalLM": "modeling_nanochat.NanoChatForCausalLM",
            "AutoTokenizer": ["tokenization_nanochat.NanoChatTokenizer", None],
        },
        "torch_dtype": "bfloat16",
        "bos_token_id": bos_token_id,
        "eos_token_id": bos_token_id,
        "pad_token_id": bos_token_id,
        "representation_layer": args.representation_layer,
        "representation_hidden_states_mode": args.representation_hidden_states_mode,
        **model_config,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "tokenizer_config.json").write_text(json.dumps({
        "auto_map": {"AutoTokenizer": ["tokenization_nanochat.NanoChatTokenizer", None]},
        "bos_token": "<|bos|>",
        "eos_token": "<|bos|>",
        "pad_token": "<|bos|>",
        "model_max_length": model_config["sequence_len"],
        "tokenizer_class": "NanoChatTokenizer",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<|bos|>",
        "eos_token": "<|bos|>",
        "pad_token": "<|bos|>",
    }, indent=2) + "\n", encoding="utf-8")
    (output_dir / "configuration_nanochat.py").write_text(CONFIGURATION_PY.lstrip(), encoding="utf-8")
    (output_dir / "modeling_nanochat.py").write_text(MODELING_PY.lstrip(), encoding="utf-8")
    (output_dir / "tokenization_nanochat.py").write_text(TOKENIZATION_PY.lstrip(), encoding="utf-8")
    shutil.copy2(tokenizer_pkl, output_dir / "tokenizer.pkl")

    state = torch.load(model_path, map_location="cpu")
    state = {k.removeprefix("_orig_mod."): v.contiguous() for k, v in state.items()}
    head_dim = model_config["n_embd"] // model_config["n_head"]
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (100000 ** (channel_range / head_dim))
    t = torch.arange(model_config["sequence_len"] * 10, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    state["cos"] = freqs.cos()[None, :, None, :].contiguous()
    state["sin"] = freqs.sin()[None, :, None, :].contiguous()
    torch.save(state, output_dir / "pytorch_model.bin")
    (output_dir / "README.md").write_text(
        "# NanoChat zh-d22 step10000 HF export\n\n"
        "This directory is a HuggingFace `trust_remote_code` export of the nanochat base checkpoint.\n"
        "Load with `AutoTokenizer.from_pretrained(path, trust_remote_code=True)` and "
        "`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`.\n",
        encoding="utf-8",
    )
    print(f"Exported HF model to {output_dir}")


if __name__ == "__main__":
    main()
