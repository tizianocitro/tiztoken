"""
Train BasicTokenizer and RegexTokenizer Tokenizers on some test data.
"""

import os
import time
from tiztoken import BasicTokenizer, RegexTokenizer

# open the test text and train a vocab of 512 tokens
text = open("tests/test_text.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

start_time = time.time()

for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)

    # writes two files in the models directory: name.model and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)

end_time = time.time()

print(f"Training took {end_time - start_time:.2f} seconds")