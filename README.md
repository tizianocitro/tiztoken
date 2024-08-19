# TizToken

A byte-level Byte Pair Encoding (BPE) algorithm for LLM tokenization. The BPE algorithm is byte-level because it runs on UTF-8 encoded strings and modern LLMs, such as GPT, Llama, and Mistral, use it to train their tokenizers.

We provided two Tokenizers (`BasicTokenizer`, `RegexTokenizer`), both of which can perform the 3 primary functions of a Tokenizer:

1. Train the tokenizer vocabulary and merges on a given text;
2. Encode from text to tokens;
3. Decode from tokens to text.

We also developed the `GPT4Tokenizer` that is a wrapper around the `RegexTokenizer`.

The files of the repo are as follows:

1. [tiztoken/base.py](tiztoken/base.py): Implements the `Tokenizer` class, which is the base class. It contains the `train`, `encode`, and `decode` stubs, save/load functionality, and there are also a few common utility functions. Do not use this class directly, use one of the class that inherit from it.
2. [tiztoken/basic.py](tiztoken/basic.py): Implements the `BasicTokenizer`, the simplest implementation of the BPE algorithm that runs directly on text.
3. [tiztoken/regex.py](tiztoken/regex.py): Implements the `RegexTokenizer` that splits the input text by a regex pattern, which is a preprocessing stage that splits up the input text by categories (letters, numbers, punctuation) before tokenization. This ensures that no merges will happen across category boundaries. This approach was introduced in the [GPT2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) ([code](https://github.com/openai/gpt-2) from OpenAI) and continues to be in use as of GPT4. This class also handles special tokens.
4. [tiztoken/gpt4.py](tiztoken/gpt4.py): Implements the `GPT4Tokenizer` and is a wrapper around the `RegexTokenizer` that reproduces the tokenization of GPT4 in the [tiktoken](https://github.com/openai/tiktoken) library. The wrapping handles some details around recovering the exact merges in the tokenizer, and the handling of some 1-byte token permutations.
5. [train.py](train.py): Trains `BasicTokenizer` and `RegexTokenizer` on the input text [tests/test_text.txt](tests/test_text.txt) and saves the vocab to disk for visualization.

## Get Started

Here is a simple example using `BasicTokenizer`:

```python
from tiztoken import BasicTokenizer

tokenizer = BasicTokenizer()

text = "aaabdaaabac"

# 256 are the byte tokens, then do 3 merges
tokenizer.train(text, 256 + 3)

print(tokenizer.encode(text))
# ---> [258, 100, 258, 97, 99]

print(tokenizer.decode([258, 100, 258, 97, 99]))
# ---> aaabdaaabac

# writes two files: example.model (for loading) and example.vocab (for viewing)
tokenizer.save("example")
```

## Testing

We use the `pytest` library for tests. All of them are in the `tests/` directory. Run `pip install pytest` and then run the tests using:

```bash
# run the tests (-v is verbose)
$ pytest -v .
```

In this way, you can be sure everything is working fine at any time.

## Training

It is possible to train the tokenizers following two possible paths.

On the one hand, If you don't want the complexity of splitting and preprocessing text with regex patterns, and you also don't need special tokens. In that case, you can use the `BasicTokenizer`. 

You can train it, and then encode and decode as follows:

```python
from tiztoken import BasicTokenizer

tokenizer = BasicTokenizer()

tokenizer.train(very_long_training_string, vocab_size=4096)

tokenizer.encode("hello world")
tokenizer.decode([1000, 2000, 3000])

tokenizer.save("mymodel")
tokenizer.load("mymodel.model")
```

On the other hand, if you want all of the above, you can use the `RegexTokenizer`. For example:

```python
from tiztoken import RegexTokenizer

tokenizer = RegexTokenizer()

tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.encode("hello world")
tokenizer.decode([1000, 2000, 3000])

tokenizer.save("mymodel")
tokenizer.load("mymodel.model")
```
**Vocabulary size**. Consider changing it as it best fits your needs.

**Special tokens**. If you need to add special tokens to your tokenizer, register them using the `register_special_tokens()` function. For example if you train with `vocab_size` of `32768`, then the first 256 tokens are raw byte tokens, the next 32768-256 are merge tokens, and after those you can add the special tokens. The last real merge token will have id of 32767 (vocab_size - 1), so your first special token should come right after that, with an id of exactly 32768. For instance:

```python
from tiztoken import RegexTokenizer

tokenizer = RegexTokenizer()

tokenizer.train(very_long_training_string, vocab_size=32768)

# you can add more tokens after this one if you need
tokenizer.register_special_tokens({"<|endoftext|>": 32768})

tokenizer.encode("<|endoftext|>hello world", allowed_special="all")
```

## TizToken GPT4 vs Tiktoken GPT4

We can verify that the `RegexTokenizer` has feature parity with the GPT4 tokenizer from [tiktoken](https://github.com/openai/tiktoken) as follows:

```python
text = "hello123!!!? (안녕하세요!) 😉"

# tiktoken
import tiktoken
# pretrained tokenizer from tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text))
# ---> [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

# tiztoken
from tiztoken import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text))
# ---> [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]
```

Run `pip install tiktoken` to execute. Basically, the `GPT4Tokenizer` is just a wrapper around `RegexTokenizer`, passing in the merges and the special tokens of GPT-4. We can also ensure the special tokens are handled correctly:

```python
text = "<|endoftext|>hello world"

# tiktoken
import tiktoken
# pretrained tokenizer from tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text, allowed_special="all"))
# ---> [100257, 15339, 1917]

# tiztoken
from tiztoken import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text, allowed_special="all"))
# ---> [100257, 15339, 1917]
```

Note that just like tiktoken, we have to explicitly declare our intent to use and parse special tokens in the call to encode. This is to avoid unintentionally tokenizing attacker-controlled data (e.g. user prompts) with special tokens. The `allowed_special` parameter can be set to `all`, `none`, or a `list of special tokens to allow`.