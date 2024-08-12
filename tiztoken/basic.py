"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, TOKENS, get_stats, merge

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= TOKENS
        num_merges = vocab_size - TOKENS

        # input text preprocessing
        # from raw bytes
        text_bytes = text.encode("utf-8")
        # to list of integers in range 0..255
        ids = list(text_bytes)

        # (int, int) -> int
        merges = {}
        # int -> bytes
        vocab = {idx: bytes([idx]) for idx in range(TOKENS)}

        # iteratively merge the most common pairs to create new tokens
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = TOKENS + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # prints if the verbose flag is set
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        # used in encode()
        self.merges = merges
        # used in decode()
        self.vocab = vocab

    def decode(self, ids):
        # given ids (list of integers), return a string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # from raw bytes
        text_bytes = text.encode("utf-8")
        # to list of integers in range 0..255
        ids = list(text_bytes)

        # given a string text, return the token ids
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                # nothing can be merged anymore
                break

            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids
