# Multilingual Word Embeddings for Indian Languages

300-dimensional FastText Skip-gram (SGNS) word embeddings trained from scratch on IndicCorpV2-5k for Gujarati, Hindi, Marathi, Punjabi, Tamil, Telugu, and Urdu.
Note: Please first remove rows after 104 from Marathi and remove 81th row from Tamil.

## 1. Folder Structure

```text
├── source_code.ipynb                  # Complete source code & inference loader
├── requirements.txt                      # Python dependencies
├── README.md                             # Setup and private evaluation guide
├── models/                               # Saved FastText checkpoints (.pt)
│   └── <language>_fasttext.pt
├── embeddings/
│   ├── <language>_fasttext_pairs.npy     # (N, 2, 300) public benchmark pair vectors
│   ├── <language>_vocabulary.json        # Word-to-ID mapping
│   └── <language>_fasttext_vocabulary_vectors.npy
├── results/                              # Evaluation CSVs and plots
└── report/
    └── NLP_Experimental_Report.pdf       # Experimental report
```

## 2. Requirements & Installation

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Optional (Linux/Colab only, needed only if re-generating t-SNE plots with native Indic script labels):

```bash
sudo apt-get update && sudo apt-get install -y fonts-noto-core
```

## 3. Evaluating on Private Data (No Retraining Required)

You do **not** need to re-run data downloading or training cells.

1. Open `source_code.ipynb` from the directory containing the `models/` folder.
2. Run **only Cell `[9]`** (*Standalone Model Loader & Arbitrary/OOV Word Embedding Inference*).
3. Use the loaded functions to generate 300-dimensional embeddings for any known or out-of-vocabulary (OOV) words, or to evaluate a private dataset file:

### A. Single Word Embedding (Known or OOV)

```python
state = load_model_checkpoint("Gujarati")
vec = generate_embedding("નવોશબ્દ", "Gujarati", loaded_state=state)
print(vec.shape)  # (300,)
```

### B. Batch of Word Pairs -> `(N, 2, 300)`

```python
pairs = [("ભારત", "દેશ"), ("અજાણ્યોશબ્દ", "ભાષા")]
pair_vectors = generate_pair_embeddings(pairs, "Gujarati")
print(pair_vectors.shape)  # (2, 2, 300)
```

### C. Evaluate a Private Benchmark File (`word1`, `word2`, `human_score`)

```python
import numpy as np
from scipy.stats import spearmanr

def evaluate_private_file(filepath: str, language: str):
    pairs, human_scores = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = [p.strip() for p in line.split("\t") if p.strip()]
            elif "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip()]
            else:
                parts = line.split()

            if len(parts) < 3:
                continue
            try:
                score = float(parts[-1])
            except ValueError:
                continue

            pairs.append((parts[0], parts[1]))
            human_scores.append(score)

    vecs = generate_pair_embeddings(pairs, language)  # shape: (N, 2, 300)

    sims = []
    for v1, v2 in vecs:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        sims.append(0.0 if (n1 == 0 or n2 == 0) else float(np.dot(v1, v2) / (n1 * n2)))

    corr, p_val = spearmanr(sims, human_scores)
    return corr, p_val, vecs

# Example usage:
# rho, p, pair_embeddings = evaluate_private_file("private_gujarati.txt", "Gujarati")
# print(f"Spearman: {rho:.4f} (p={p:.2e})")
```

> Supported language names: `"Gujarati"`, `"Hindi"`, `"Marathi"`, `"Punjabi"`, `"Tamil"`, `"Telugu"`, `"Urdu"` (case-insensitive).