# TizToken

A byte-level Byte Pair Encoding (BPE) algorithm for LLM tokenization. The BPE algorithm is byte-level because it runs on UTF-8 encoded strings and modern LLMs, such as GPT, Llama, and Mistral, use it to train their tokenizers.

We provided two Tokenizers (`BasicTokenizer`, `RegexTokenizer`), both of which can perform the 3 primary functions of a Tokenizer:

1. Train the tokenizer vocabulary and merges on a given text;
2. Encode from text to tokens;
3. Decode from tokens to text.