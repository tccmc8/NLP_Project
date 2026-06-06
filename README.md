# NLP Project — Multi-Class Text Classification on 20-Newsgroups

> **NALAPRO Assignment** · A systematic comparison of five NLP approaches to text classification, from a simple neural network to QLoRA fine-tuning of a large language model.

---

## Task Synopsis

This project was set as part of the NALAPRO module. The task requires building and comparing multiple approaches to **multi-class text classification** on the **20-Newsgroups dataset**, progressing from classical methods to state-of-the-art large language model techniques.

### Dataset

The [20-Newsgroups dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) consists of approximately **18,000 Usenet posts** distributed across **20 topically distinct categories** (e.g. `comp.graphics`, `rec.sport.hockey`, `sci.space`, `talk.politics.guns`). It is loaded directly via `sklearn.datasets.fetch_20newsgroups` with headers, footers, and quotes removed to avoid trivial classification shortcuts.

---

## Notebooks

The project is split into five notebooks, each self-contained and runnable independently on Google Colab. All notebooks mount Google Drive to `MyDrive/NALAPRO/` and persist results as CSVs for cross-part comparison.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `Part1_SimpleNN.ipynb` | Two-layer feedforward neural network with ReLU, tested across four input representations: Word2Vec (1 epoch), Word2Vec (20 epochs), TF-IDF (20k vocab), and pre-trained GloVe |
| 2 | `Part2_BERT_FineTune.ipynb` | BERT-base fine-tuning with three strategies: full fine-tuning (all 110M parameters), linear probe (frozen encoder, head only), and partial freeze (top 2 encoder layers + head) |
| 3 | `Part3_BERT_MLM_then_Classify.ipynb` | Two-stage training: domain-adaptive masked language model (MLM) pre-training on the 20-Newsgroups corpus, followed by classification fine-tuning — following Gururangan et al. (2020) |
| 4 | `Part4_Llama3_Prompting.ipynb` | Inference-only classification using Llama-3-8B-Instruct via the Groq API; no gradient updates — tested under zero-shot and 1-shot-per-class (k=1 few-shot) conditions |
| 5 | `Part5_Llama3_QLoRA.ipynb` | *(Bonus)* QLoRA fine-tuning of a Llama-3 model using 4-bit NF4 quantisation and LoRA adapters; trains only ~0.1–1% of parameters while keeping the base model frozen |

---

## Run Order

The notebooks are designed to run **in order** (1 → 2 → 3 → 4 → 5), as each part saves a results CSV that later parts load for cross-part comparison charts. Parts 4 and 5 can be run independently if the prior CSVs are not available — the comparison charts will simply skip missing data.

```
Part1_SimpleNN.ipynb          →  saves  part1_results.csv
Part2_BERT_FineTune.ipynb     →  saves  part2_results.csv
Part3_BERT_MLM_then_Classify.ipynb  →  saves  part3_results.csv
Part4_Llama3_Prompting.ipynb  →  saves  part4_results_all.csv
Part5_Llama3_QLoRA.ipynb      →  saves  part5_results.csv
```

> All CSVs and plots are saved to `MyDrive/NALAPRO/` on Google Drive so they persist across Colab sessions.

---

## 📊 Results

Results are reported on the held-out test split of 20-Newsgroups. Part 4 and 5 use a stratified 400-document evaluation subset (20 per class) for tractable inference-time evaluation.

| Part | Method | Test Accuracy | Test Macro-F1 |
|------|--------|:-------------:|:-------------:|
| 1 | Simple NN — Word2Vec (1 epoch) | 0.1442 | 0.1069 |
| 1 | Simple NN — Word2Vec (20 epochs) | 0.5875 | 0.5712 |
| 1 | Simple NN — TF-IDF (20k vocab) | 0.6868 | 0.6798 |
| 1 | Simple NN — GloVe pre-trained | 0.5722 | 0.5519 |
| 2 | BERT — Full fine-tuning | 0.7283 | 0.7093 |
| 2 | BERT — Linear probe | 0.3873 | 0.3337 |
| 2 | BERT — Partial freeze | 0.7095 | 0.6880 |
| 3 | BERT + Domain-Adaptive MLM | 0.7357 | 0.7157 |
| 4 | Llama-3-8B — Zero-shot | 0.5 | 0.4937 |
| 4 | Llama-3-8B — Few-shot (k=1) | 0.49 | 0.5089 |
| 5 | Llama-3 — QLoRA fine-tuned | 0.63 | 0.6808 |

---

## Setup & Usage

### Requirements

- A **Google account** (for Google Colab and Google Drive)
- A **Groq API key** for Part 4 (free at [console.groq.com](https://console.groq.com))
- A **Hugging Face account** for Part 5 if using a gated Llama model

### Google Colab (Recommended)

All notebooks are designed for Google Colab. Each notebook installs its own dependencies via `%pip install` in the first cell — no local setup required.

**GPU Runtime:**

| Runtime | Availability | Recommended for |
|---------|-------------|-----------------|
| T4 GPU (16 GB) | Free tier | Parts 1–5 (all notebooks) |
| A100 GPU (40 GB) | Colab Pro | Parts 2, 3, 5 for faster training |

To set the runtime: `Runtime → Change runtime type → GPU (T4 or A100)`

> ⚠️ **Part 5 (QLoRA) requires a GPU.** The notebook will raise an error if no CUDA device is detected. It will not run on CPU or Apple Silicon.

Note: Colab Pro is not necessary A100 GPU is much faster than T4 GPU, both work but it depends on the time constraints.

### Steps

1. Clone or download this repository
2. Upload the notebooks to Google Colab (or open directly from GitHub)
3. Run notebooks in order: Part 1 → Part 2 → Part 3 → Part 4 → Part 5
4. For **Part 4**: add your Groq API key to Colab Secrets (`Tools → Secrets → GROQ_API_KEY`)
5. For **Part 5** with gated Llama models: add your HuggingFace token to Colab Secrets (`HF_TOKEN`)

### Google Drive

Each notebook mounts Google Drive on first run and saves all outputs to `MyDrive/NALAPRO/`. This ensures results persist between Colab sessions.

---

## Technologies Used

| Tool / Library | Purpose |
|----------------|---------|
| `PyTorch` | Neural network training loop (Part 1), underlying compute engine (Parts 2–5) |
| `scikit-learn` | 20-Newsgroups dataset loading, TF-IDF vectorisation, evaluation metrics, confusion matrix |
| `nltk` | Tokenisation, stop-word removal, lemmatisation (Part 1 preprocessing) |
| `gensim` / `gensim-data` | Word2Vec training and pre-trained GloVe vector loading (Part 1) |
| `transformers` (Hugging Face) | Pre-trained BERT, `AutoModelForSequenceClassification`, `Trainer`, `AutoModelForMaskedLM`, Llama-3 model loading |
| `datasets` (Hugging Face) | `Dataset` / `DatasetDict` pipeline for Trainer-compatible data loading |
| `evaluate` (Hugging Face) | Accuracy and macro-F1 metric computation |
| `peft` (Hugging Face) | LoRA adapter configuration via `LoraConfig` and `get_peft_model` (Part 5) |
| `bitsandbytes` | 4-bit NF4 quantisation via `BitsAndBytesConfig` (Part 5) |
| `trl` | Supervised fine-tuning loop via `SFTTrainer` (Part 5) |
| `Groq API` | Llama-3-8B-Instruct inference for zero-shot and few-shot prompting (Part 4) |
| `Llama-3` (Meta) | Large language model used for prompting (Part 4) and QLoRA fine-tuning (Part 5) |
| `matplotlib` / `seaborn` | Training curves, confusion matrices, per-class F1 bar charts, weight distribution plots |
| `pandas` | Results tables and cross-part comparison summaries |
| `wandb` | Optional experiment tracking (disabled by default via `report_to="none"`) |
| `Google Colab` | Cloud GPU environment (T4 / A100) |
| `Google Drive` | Persistent storage for CSVs, plots, and model checkpoints across sessions |

### AI Assistance

**Claude (Anthropic)** was used as a coding and debugging tool throughout this project — assisting with code scaffolding, error diagnosis, and explanation of library APIs. All code has been read, understood, and verified by the author. All AI assistance is declared in the tools table at the top of each notebook, in compliance with the NALAPRO academic integrity guidelines.

---

## Key References

- Vaswani et al. (2017) — *Attention Is All You Need*
- Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers*
- Gururangan et al. (2020) — *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*
- Brown et al. (2020) — *Language Models are Few-Shot Learners*
- Dettmers et al. (2023) — *QLoRA: Efficient Finetuning of Quantized LLMs*
- Alammar & Grootendorst (2024) — *Hands-On Large Language Models*, O'Reilly
