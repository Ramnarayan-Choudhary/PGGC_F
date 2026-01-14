# MCQ Baseline Methods - Data and Results

This folder contains pre-generated MCQ evaluation results from various privacy-preserving baseline methods compared against DP-FUSION in the paper.

## 📁 Folder Structure

```
baselines/
├── README_BASELINES.md             # This file
├── dp_decoding/                    # DP-Decoding baseline results
│   ├── dp_decoding_lambda_0.1.json     # Generated passages (λ=0.1)
│   ├── dp_decoding_lambda_0.5.json     # Generated passages (λ=0.5)
│   ├── dp_decoding_lambda_0.75.json    # Generated passages (λ=0.75)
│   ├── dp_decoding_lambda_0.9.json     # Generated passages (λ=0.9)
│   └── dp_decoding_lambda_*_eval.json  # Evaluation results
├── baseline_paraphrases/           # Public/Private baseline paraphrases
│   ├── baseline_paraphrases.json           # Standard paraphrase prompt
│   └── baseline_paraphrases_privacy_in_prompt.json  # Privacy-aware prompt
└── rewriting/                      # DP-FUSION paraphrase outputs (beta sweep)
    ├── output_beta_0.001.json
    ├── output_beta_0.01.json
    ├── output_beta_0.1.json
    ├── output_beta_0.5.json
    ├── output_beta_1.0.json
    ├── output_beta_2.5.json
    ├── output_beta_5.0.json
    └── output_beta_10.0.json
```

## 🔬 Baseline Methods

### 1. DP-Decoding (Baseline)

**Method**: Differentially private text generation via token-level noise injection
**Paper Reference**: "Differentially Private Language Models via Token-Level Noise Injection"

**Parameters**: λ (lambda) - controls privacy level
- **λ = 0.1**: Highest privacy, lowest utility
- **λ = 0.5**: Medium privacy
- **λ = 0.75**: Lower privacy
- **λ = 0.9**: Lowest privacy, highest utility

**Files**:
- `dp_decoding_lambda_X.json`: Generated paraphrases for each λ value
- `dp_decoding_lambda_X_eval.json`: MCQ evaluation results

**Data Format** (`dp_decoding_lambda_X.json`):
```json
[
  {
    "passage": ["line1", "line2", ...],
    "private_entities": [...],
    "redacted_text": "...privatized passage...",
    "metadata": {
      "doc_id": "doc_0979",
      "question": "...",
      "options": [...],
      "answer_index": 2
    }
  }
]
```

**Paper Results** (Table 16):
| λ | Accuracy (%) | ASR (%) |
|---|--------------|---------|
| 0.1 | 23 | 15.67 |
| 0.9 | 70 | 66.00 |

### 2. DP-Prompt (Baseline)

**Method**: Privacy via prompt engineering (instructing model to avoid private information)

**Parameters**:
- **w** (clip norm): Gradient clipping threshold (5, 50)
- **T** (temperature): Sampling temperature (0.75, 1.75)

**Status**: ⚠️ **Not included in this package**
- DP-Prompt results are referenced in the paper (Table 16) but the generated passages are not in this repository
- The method uses prompt-based privatization without modifying the model

**Paper Results** (Table 16):
| Parameters | Accuracy (%) | ASR (%) |
|------------|--------------|---------|
| w=5, T=0.75 | 31 | 26.67 |
| w=5, T=1.75 | 32 | 17.33 |
| w=50, T=0.75 | 90 | 56.67 |
| w=50, T=1.75 | 33 | 28.67 |

### 3. Baseline Paraphrases (No DP)

**Method**: Standard paraphrasing without differential privacy

**Files**:
- `baseline_paraphrases.json`: Standard paraphrase prompt
- `baseline_paraphrases_privacy_in_prompt.json`: Privacy instruction in prompt

**Data Format**:
```json
[
  {
    "passage": ["line1", "line2", ...],
    "private_entities": [...],
    "public_paraphrase": "...paraphrase with entities redacted...",
    "private_paraphrase": "...paraphrase with all entities...",
    "metadata": {
      "doc_id": "doc_0979",
      "question": "...",
      "options": [...]
    }
  }
]
```

**Purpose**:
- **PUBLIC paraphrase**: All entities replaced with "_" → tests minimum utility baseline
- **PRIVATE paraphrase**: All entities visible → tests maximum utility baseline (no privacy)

**Paper Results** (Table 16):
| Variant | Accuracy (%) | ASR (%) |
|---------|--------------|---------|
| No DPI (Original) | 98 | 62.70 |
| No DPI (NER) | 34 | 27.70 |

### 4. DP-FUSION Paraphrases (Rewriting)

**Method**: DP-FUSION with paraphrase-style generation

**Files**: `output_beta_X.json` for β ∈ {0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0}

**Data Format**:
```json
[
  {
    "passage": ["line1", "line2", ...],
    "private_entities": [...],
    "redacted_text": "...DP-FUSION privatized passage...",
    "metadata": {
      "doc_id": "doc_0979",
      "question": "...",
      "options": [...],
      "answer_index": 2
    }
  }
]
```

**Paper Results** (Table 16):
| αβᵢ | Accuracy (%) | ASR (%) |
|-----|--------------|---------|
| 0.001 | 36 | 28.00 |
| 0.01 | 37 | 26.00 |
| 0.1 | 38 | 29.30 |
| 5.0 | 60 | 45.70 |
| 10.0 | 85 | 56.30 |

## 📊 File Format Details

### DP-Decoding Files (`dp_decoding_lambda_X.json`)
Each entry contains:
- `passage`: Original text split into lines
- `private_entities`: Entity annotations
- `redacted_text`: DP-Decoding privatized text
- `metadata`: MCQ question, options, answer

### Evaluation Files (`*_eval.json`)
Results after running MCQ evaluation:
```json
{
  "config_name": {
    "accuracy": 0.37,
    "correct": 37,
    "total": 100,
    "results": [
      {
        "doc_id": "doc_0979",
        "predicted_answer": "C",
        "correct_answer": "C",
        "is_correct": true,
        "generated_text": "..."
      }
    ]
  }
}
```

### Baseline Paraphrase Files
Each entry has TWO paraphrases:
- `public_paraphrase`: Generated from redacted text (entities = "_")
- `private_paraphrase`: Generated from full text (all entities visible)

This allows comparing utility at two privacy extremes.

## 🔧 How to Use These Files

### 1. Evaluate Pre-generated Passages

Use `evaluate_sanitized_passages.py` from the `code/` folder:

```bash
# Evaluate DP-Decoding
python code/evaluate_sanitized_passages.py \
  --dp_methods \
  --dp_methods_file data/baselines/dp_decoding/dp_decoding_lambda_0.5.json \
  --output_file dp_decoding_lambda_0.5_eval.json \
  --gpu 0 \
  --hf_token YOUR_TOKEN

# Evaluate baseline paraphrases
python code/evaluate_sanitized_passages.py \
  --baselines \
  --baseline_file data/baselines/baseline_paraphrases/baseline_paraphrases.json \
  --output_file baseline_eval.json \
  --gpu 0 \
  --hf_token YOUR_TOKEN
```

### 2. Generate New Baseline Paraphrases

```bash
python code/baseline_paraphrasing.py \
  --input_file data/input_converted_mcq.json \
  --start 0 --end 100 \
  --gpu 0 \
  --output_file new_baseline_paraphrases.json \
  --hf_token YOUR_TOKEN
```

### 3. Compare Methods

All files use the same format, making it easy to:
- Compare accuracy across methods
- Analyze which entity types are preserved/leaked
- Study privacy-utility tradeoffs

## 📖 Paper Context

These baselines appear in:
- **Table 16**: Performance comparison across privacy-preserving methods
- **Figure 4**: Privacy-utility tradeoff curves
- **Section 5.3**: Comparison with baselines

**Key Finding**: DP-FUSION achieves better privacy-utility tradeoffs than all baselines. At comparable privacy levels (ASR ~26-27%), DP-FUSION provides higher utility (37-38% accuracy) than:
- DP-Decoding (23% at λ=0.1)
- DP-Prompt (31-33% at various settings)
- No DPI-NER (34%)

## 🔍 Missing Data: DP-Prompt

**Status**: DP-Prompt passages are **not included** in this repository.

**Why**: DP-Prompt is a prompt-based method that may have been:
- Implemented externally
- Generated using a different pipeline
- Or the passages were not saved (only evaluation results reported)

**What's Available**: Paper results (Table 16) show DP-Prompt accuracy and ASR

**To Reproduce**: Would need to implement DP-Prompt method or obtain original passages from authors

## 📝 Citation

If using these baseline comparisons, cite both DP-FUSION and the original baseline papers:

```bibtex
@inproceedings{dpfusion2025,
  title={DP-FUSION: Differentially Private Text Generation via Fusion of Language Models},
  booktitle={Under review at ICLR 2026},
  year={2025}
}

% Cite original DP-Decoding paper
% Cite original DP-Prompt paper
```

---

**Summary**: This folder contains pre-generated privatized passages from DP-Decoding, baseline paraphrasing, and DP-FUSION methods, allowing for direct comparison of downstream MCQ task performance across different privacy-preserving approaches.
