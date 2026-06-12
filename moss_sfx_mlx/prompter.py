"""MLX-side transpose of moss_soundeffect_v2/diffsynth/prompters/wan_prompter.py.

Text processing is framework-free and copied verbatim; the only change is the
encoder call returning an mx.array. Pad-position embeddings are zeroed exactly
as upstream (the DiT cross-attn is mask-free; zeroing IS the masking).
"""

import html
import string

import ftfy
import mlx.core as mx
import regex as re


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, 'whitespace', 'lower', 'canonicalize')
        from transformers import AutoTokenizer

        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop('return_mask', False)

        _kwargs = {'return_tensors': 'np'}
        if self.seq_len is not None:
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
        _kwargs.update(**kwargs)

        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        if return_mask:
            return ids.input_ids, ids.attention_mask
        return ids.input_ids

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        return text


class WanPrompter:

    def __init__(self, tokenizer_path=None, text_len=512):
        self.text_len = text_len
        self.text_encoder = None
        self.fetch_tokenizer(tokenizer_path)

    def fetch_tokenizer(self, tokenizer_path=None):
        if tokenizer_path is not None:
            self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=self.text_len, clean='whitespace')

    def fetch_models(self, text_encoder=None):
        self.text_encoder = text_encoder

    def encode_prompt(self, prompt, positive=True):
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        seq_lens = mask.sum(axis=1).astype(int)
        prompt_emb = self.text_encoder(mx.array(ids), mx.array(mask))
        # Zero the tail of each sample by its valid length so a shorter sample
        # in a batch is not truncated to the longest sample's length.
        # (mx arrays don't support sliced in-place assignment across rows the
        # torch way; build a mask instead — same values.)
        positions = mx.arange(prompt_emb.shape[1])[None, :, None]
        valid = positions < mx.array(seq_lens)[:, None, None]
        prompt_emb = prompt_emb * valid
        return prompt_emb
