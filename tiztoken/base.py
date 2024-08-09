"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
"""
import unicodedata

# a few helper functions useful for BasicTokenizer and RegexTokenizer

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    # iterate consecutive elements
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    merged_ids = []
    i = 0
    while i < len(ids):
        # if not at the very last position and the pair matches,
        # replace the pair with the new idx
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            merged_ids.append(idx)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters because they distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        # this character is ok
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            # escape
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# the base Tokenizer class
VERSION = "tiztoken v1"
TOKENS = 256

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        # (int, int) -> int
        self.merges = {}
        # str, for example like in the regex tokenizer
        self.pattern = ""
        # str -> int, e.g. {'<|endoftext|>': 100257}
        self.special_tokens = {}
        # int -> bytes
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(TOKENS)}

        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode("utf-8")

        return vocab

    def save(self, file_name):
        """
        Saves two files: file_name.vocab and file_name.model
        This is inspired by the sentencepiece library's model saving:
        - model file is the important one, used for the load() operation
        - vocab file is a pretty printed version for human inspection
        """
        # write the model to be used in load()
        model_file = file_name + ".model"
        with open(model_file, 'w') as f:
            # write the version (compatibility) and pattern
            f.write(f"{VERSION}\n")
            f.write(f"{self.pattern}\n")

            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special_token, idx in self.special_tokens.items():
                f.write(f"{special_token} {idx}\n")

            # write the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # write the vocab for the human inspection
        vocab_file = file_name + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. So we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, so just print it
                    # this should correspond to the first 256 tokens, the bytes
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = TOKENS
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == VERSION

            # read the pattern
            self.pattern = f.readline().strip()

            # read the special tokens
            num_special_tokens = int(f.readline().strip())

            for _ in range(num_special_tokens):
                special_token, special_token_idx = f.readline().strip().split()
                special_tokens[special_token] = int(special_token_idx)

            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        # assign the loaded merges and special tokens
        # and build the vocab from the them
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
