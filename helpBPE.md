## Byte Pair Encoding (BPE)

The byte pair encoding method is a famous tokenization algorithm, used in several different models such as GPT-2/4, LLaMA 3, DeepSeek-V3, and DeepSeek-R1.

This tokenizer operates at the byte level, enabling it to handle a wide range of characters and languages without relying on a **predefined vocabulary**. The byte-level BPE approach is particularly effective for multilingual models, as it can seamlessly process text from various languages, including those with complex character sets.

The goal of the BPE tokenization algorithm is to build a vocabulary of commonly occurring subwords. The BPE algorithm was originally described in 1994: “A New Algorithm for Data Compression” by Philip Gage [1].

The algorithm is summarized as follows:

- Begin with a corpus where each word is split into individual characters.
- Iteratively identify the most frequent adjacent pair of symbols (characters or subwords).
- Merge this pair into a new token and record it in a vocabulary (lookup table).
- Repeat the merging process until a predefined vocabulary size is reached or no frequent pairs remain.

The resulting vocabulary allows for efficient encoding of rare and unseen words by decomposing them into known subword units. Decoding reverses the process by replacing tokens with their original pairs using the lookup table.

BPE is a compression-based subword tokenization algorithm. It merges frequent pairs of characters or tokens to form new tokens, reducing overall vocabulary size while retaining meaning.

### BPE Example

Assume the input text is: `low lower lowest`

**Step 1:** Split all words into characters (with an end-of-word marker `_`):

- `l o w _`
- `l o w e r _`
- `l o w e s t _`

**Step 2:** Count all symbol pairs and find the most frequent:

- Most frequent pair: `l o`

**Step 3:** Merge `l o` into a new symbol:

- `lo w _`
- `lo w e r _`
- `lo w e s t _`

**Step 4:** Repeat the process:

- Merge `lo w` ⇒ `low _`, `low e r _`, `low e s t _`
- Merge `low e` ⇒ `lowe r _`, `lowe s t _`
- Merge `lowe r` ⇒ `lower _`, `lower s t _`

**Step 5:** Stop when no frequent pairs remain or when desired vocabulary size is reached.

### Final Vocabulary

Possible vocabulary:

l, o, w, lo, low, e, r, s, t, lowe, lower, _



### Decoding

To decode, reverse the merges using the lookup table step by step.

---

### References

[1] Philip Gage. *A New Algorithm for Data Compression*. 1994. [Link](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)



