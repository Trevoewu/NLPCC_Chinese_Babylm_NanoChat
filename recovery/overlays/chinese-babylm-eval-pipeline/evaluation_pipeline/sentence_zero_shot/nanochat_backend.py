from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput


def _ensure_nanochat_on_path() -> None:
    repo = os.environ.get("NANOCHAT_REPO", "/home/trevor/babyllm")
    if repo not in sys.path:
        sys.path.insert(0, repo)


class NanoChatTokenizerAdapter:
    """Minimal tokenizer facade for Chinese zero-shot causal scoring."""

    def __init__(self) -> None:
        _ensure_nanochat_on_path()
        from nanochat.tokenizer import get_tokenizer

        self.tokenizer = get_tokenizer()
        self.bos_token_id = self.tokenizer.get_bos_token_id()
        self.eos_token_id = self.bos_token_id
        self.pad_token_id = self.bos_token_id
        self.cls_token_id = self.bos_token_id
        self.mask_token_id = None
        self.pad_token = "<|bos|>"
        self.eos_token = "<|bos|>"
        self.bos_token = "<|bos|>"
        self.all_special_ids = [self.bos_token_id]
        self.model_max_length = 1024
        self.is_fast = False

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        prepend = "<|bos|>" if add_bos else None
        return self.tokenizer.encode(text, prepend=prepend)

    def _encode_split_words(self, words: list[str], add_bos: bool) -> tuple[list[int], list[int]]:
        ids = [self.bos_token_id] if add_bos else []
        special_mask = [1] if add_bos else []
        for word in words:
            word_ids = self.tokenizer.encode(word)
            ids.extend(word_ids)
            special_mask.extend([0] * len(word_ids))
        return ids, special_mask

    def _encode_text(self, text, add_bos: bool, is_split_into_words: bool) -> tuple[list[int], list[int]]:
        if is_split_into_words:
            if not isinstance(text, list):
                raise TypeError("is_split_into_words=True expects a list of words")
            return self._encode_split_words(text, add_bos=add_bos)
        if isinstance(text, tuple):
            text = "\n".join(str(part) for part in text)
        ids = self.encode(str(text), add_bos=add_bos)
        special_mask = [0] * len(ids)
        if add_bos and special_mask:
            special_mask[0] = 1
        return ids, special_mask

    def _batch_encode(
        self,
        texts,
        return_tensors: str | None,
        padding: bool,
        truncation: bool,
        max_length: int | None,
        add_special_tokens: bool,
        is_split_into_words: bool,
        return_special_tokens_mask: bool,
    ):
        if isinstance(texts, tuple):
            examples = [texts]
        elif isinstance(texts, list) and is_split_into_words and all(isinstance(x, str) for x in texts):
            examples = [texts]
        elif isinstance(texts, list):
            examples = texts
        else:
            examples = [texts]

        encoded = []
        masks = []
        special_masks = []
        for example in examples:
            ids, special_mask = self._encode_text(example, add_bos=add_special_tokens, is_split_into_words=is_split_into_words)
            if truncation and max_length is not None:
                ids = ids[:max_length]
                special_mask = special_mask[:max_length]
            encoded.append(ids)
            masks.append([1] * len(ids))
            special_masks.append(special_mask)

        if padding:
            pad_to = max(len(ids) for ids in encoded) if encoded else 0
            if max_length is not None and padding == "max_length":
                pad_to = max_length
            for ids, mask, special_mask in zip(encoded, masks, special_masks):
                pad_len = max(0, pad_to - len(ids))
                ids.extend([self.pad_token_id] * pad_len)
                mask.extend([0] * pad_len)
                special_mask.extend([1] * pad_len)

        out = BatchEncodingAdapter({
            "input_ids": encoded,
            "attention_mask": masks,
        })
        if return_special_tokens_mask:
            out["special_tokens_mask"] = special_masks

        if return_tensors == "pt":
            out = BatchEncodingAdapter({key: torch.tensor(value, dtype=torch.long) for key, value in out.items()})
        elif len(examples) == 1:
            out = BatchEncodingAdapter({key: value[0] for key, value in out.items()})
        return out

    def __call__(self, *args, text: str | None = None, **kwargs) -> dict[str, list[int] | list[tuple[int, int]]]:
        if args:
            text = args[0]
        if text is None:
            text = kwargs.get("text", "")
        if "text_pair" in kwargs and kwargs["text_pair"] is not None:
            text = (text, kwargs["text_pair"])

        return_tensors = kwargs.get("return_tensors")
        padding = kwargs.get("padding", False)
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length")
        add_special_tokens = kwargs.get("add_special_tokens", True)
        is_split_into_words = kwargs.get("is_split_into_words", False)
        return_special_tokens_mask = kwargs.get("return_special_tokens_mask", False)

        if return_tensors is not None or padding or isinstance(text, list) or isinstance(text, tuple) or is_split_into_words:
            return self._batch_encode(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
                is_split_into_words=is_split_into_words,
                return_special_tokens_mask=return_special_tokens_mask,
            )

        add_special_tokens = kwargs.get("add_special_tokens", True)
        ids = self.encode(text, add_bos=add_special_tokens)
        out = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        if kwargs.get("return_offsets_mapping", False):
            out["offset_mapping"] = [(0, 0)] * len(ids)
        return out

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        del pair
        return 1

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return [self.tokenizer.decode([token_id]) for token_id in ids]


class BatchEncodingAdapter(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device):
        return BatchEncodingAdapter({
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.items()
        })


class NanoChatCausalModel(nn.Module):
    """Transformers-like wrapper around a nanochat base checkpoint."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        _ensure_nanochat_on_path()
        from nanochat.checkpoint_manager import load_model

        model_tag = os.environ.get("NANOCHAT_MODEL_TAG", "zh-d22-2gpu")
        step_text = os.environ.get("NANOCHAT_MODEL_STEP", "")
        step = int(step_text) if step_text else None
        self.model, self.tokenizer, self.meta = load_model(
            "base",
            device,
            phase="eval",
            model_tag=model_tag,
            step=step,
        )
        self.bos_token_id = self.tokenizer.get_bos_token_id()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        del attention_mask, kwargs
        if input_ids.size(1) == 1:
            # nanochat's training forward expects at least two positions because
            # of the smear block. The first-position logits are unchanged by a
            # dummy future BOS under causal attention, so slice them back out.
            filler = torch.full_like(input_ids, self.bos_token_id)
            logits = self.model(torch.cat([input_ids, filler], dim=1))
            logits = logits[:, :1, :]
        else:
            logits = self.model(input_ids)
        return {"logits": logits}


class NanoChatBackboneModel(nn.Module):
    """Backbone wrapper returning final hidden states for fine-tune/CogBench."""

    def __init__(self, device: torch.device | str) -> None:
        super().__init__()
        device = torch.device(device)
        _ensure_nanochat_on_path()
        from nanochat.checkpoint_manager import load_model

        model_tag = os.environ.get("NANOCHAT_MODEL_TAG", "zh-d22-2gpu")
        step_text = os.environ.get("NANOCHAT_MODEL_STEP", "")
        step = int(step_text) if step_text else None
        self.model, self.tokenizer, self.meta = load_model(
            "base",
            device,
            phase="eval",
            model_tag=model_tag,
            step=step,
        )
        model_config = self.meta["model_config"]
        self.config = SimpleNamespace(
            hidden_size=model_config["n_embd"],
            max_position_embeddings=model_config["sequence_len"],
            is_encoder_decoder=False,
        )
        self.bos_token_id = self.tokenizer.get_bos_token_id()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        del attention_mask, kwargs
        captured = {}

        def hook(_module, hook_inputs, _output):
            captured["last_hidden_state"] = hook_inputs[0]

        handle = self.model.lm_head.register_forward_hook(hook)
        added_filler = False
        try:
            if input_ids.size(1) == 1:
                filler = torch.full_like(input_ids, self.bos_token_id)
                _ = self.model(torch.cat([input_ids, filler], dim=1))
                added_filler = True
            else:
                _ = self.model(input_ids)
        finally:
            handle.remove()

        hidden = captured["last_hidden_state"]
        if added_filler:
            hidden = hidden[:, :1, :]
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=(hidden,))


def is_nanochat_model_path(model_path_or_name: str) -> bool:
    return model_path_or_name.startswith("nanochat")
