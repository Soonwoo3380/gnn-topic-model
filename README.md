# Graph Neural Network-Based Topic Model Integrating Probabilistic and Embedding Information of Words

> 🎓 **Master's Thesis (Dec 2025)**
> - **Author:** Soonwoo Kim 
> - **Supervisor:** Prof. Suhyeon Kim
> - **Institution:** Department of Data Science, Kyungpook National University
> - **Link:** https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=5196fd2e578fffceffe0bdc3ef48d419


---

## Motivation
This project originated from a fundamental question: **can heterogeneous data of different natures be jointly leveraged?** Data constructed for a specific task inherently possess properties well-suited to addressing that task's objectives. This inquiry was further refined into a more concrete question: whether such properties could be effectively exploited across heterogeneous data sources. Motivated by this curiosity, we represent these relationships as a graph structure and apply the proposed framework to topic modeling.

---

## Overview

This repository contains the full implementation of a novel topic modeling framework that integrates **LDA**, **BERTopic**, and a **Deep Modularity Networks (DMoN)** to overcome the limitations of each standalone model.

- LDA captures probabilistic co-occurrence structure but lacks semantic similarity.
- BERTopic captures contextual coherence but lacks interpretability of topic-word relations.
- Our method fuses probabilistic and embedding informations into a keyword graph and applies DMoN clustering to extract structurally and semantically coherent topics in an end-to-end manner.

---

## Methodology

### Stage 1 — Topic Model Output Extraction
- LDA produces a topic-word probability matrix (K words × T topics).
- BERTopic (via SBERT, `paraphrase-multilingual-mpnet-base-v2`) produces 384-dim word embeddings.

### Stage 2 — Keyword Graph Construction
| Edge Connections | Node Feature |
| :---: | :---: |
| ![Edge Connections](figures/2.png) | ![Node Feature](figures/3.png) |

- **Strong edges**: Top-K keywords per LDA topic form dense intra-topic connections (edge weight = 1).
- **Weak edges**: Each non-core word connects to its most similar core word via cosine similarity (0 < weight < 1).

### Stage 3 — DMoN-based Clustering
- Node features = SBERT word embeddings (384-dim)
- A 3-layer GCN encoder performs message passing over the keyword graph.
- DMoN's modularity loss + collapse regularization identifies structurally stable topic clusters.

![Overall Process](figures/4.png)

---
## Data

### Data Collection
News articles were collected from [BIG KINDS](https://www.bigkinds.or.kr), 
a large-scale Korean news database operated by the Korea Press Foundation.

| Item | Detail |
|------|--------|
| Query | "생성형 AI 영어교육" (English education using generative AI) |
| Period | January 2023 – February 2025 (23 months) |
| # of Articles | 1,015 |
| # of Media Outlets | 88 |
| Variables Used | Keywords (nouns extracted from article bodies) |

### Data Preprocessing
The following steps were applied to the extracted keyword vocabulary:

1. **Proper noun removal** – 3,100 proper nouns (names, institutions, brands, etc.) excluded
2. **Frequency filtering (Zipf's Law)** – Words appearing fewer than 10 or more than 1,000 times removed
3. **Language filtering** – Only Korean words retained; English, Chinese characters, numerals, and symbols excluded

| Stage | Vocabulary Size |
|-------|----------------|
| Raw unique words | 27,182 |
| After preprocessing | **2,597** |

---

## Key Results
| Model | C<sub>w2v</sub> | C<sub>v</sub> | Modularity |
|---|---|---|---|
| LDA | 0.5846 | 0.4391 | 0.1910 |
| BERTopic | 0.6779 | 0.4754 | 0.2347 |
| **Ours (DMoN)** | **0.7220** | **0.5621** | **0.3454** |

![Performance Comparison](figures/5.png)

---

## Ablation Study

### Effect of Top-K Setting

The number of top words per topic (K) controls graph density. K was varied from 10 to 50 in steps of 10, with the number of topics fixed at 30.

| Top-K | Total Edges | Strong | Weak | Modularity | C<sub>w2v</sub> | C<sub>v</sub> |
|-------|-------------|--------|------|------------|--------|-----|
| 10 | 3,198 | 1,145 | 2,053 | 0.5843 | 0.5405 | 0.7089 |
| 20 | 6,529 | 4,561 | 1,968 | 0.4482 | 0.5575 | 0.7160 |
| **30** | **11,960** | **10,075** | **1,885** | **0.3454** | **0.5621** | **0.7220** |
| 40 | 19,113 | 17,300 | 1,813 | 0.2823 | 0.5595 | 0.7268 |
| 50 | 27,916 | 26,169 | 1,747 | 0.1866 | 0.5500 | 0.7218 |

As K increases, strong edges dominate (35.8% → 93.7%), homogenizing node connectivity and blurring cluster boundaries — modularity drops ~68%. **K=30** achieves the best balance across all three metrics.

---

### Effect of Edge Configuration

Three edge configurations were tested to verify the contribution of the strong+weak design.

| Configuration | Total Edges | Strong | Weak | Modularity | C<sub>w2v</sub> | C<sub>v</sub> |
|---------------|-------------|--------|------|------------|--------|-----|
| Weak-only | 1,885 | 0 | 1,885 | 0.6437 | 0.5271 | 0.6992 |
| **Strong+Weak (Proposed)** | **11,960** | **10,075** | **1,885** | **0.3454** | **0.5621** | **0.7220** |
| Strong-only | 10,075 | 10,075 | 0 | 0.2960 | 0.5696 | 0.7200 |

- **Weak-only**: Highest modularity (clearest cluster boundaries) but lowest semantic coherence — peripheral words lack sufficient semantic binding.
- **Strong-only**: Higher C<sub>v</sub> and C<sub>w2v</sub> but blurred cluster boundaries due to excessive edge density.
- **Strong+Weak (Proposed)**: Strong edges anchor the semantic core of each topic; weak edges link peripheral words and expand the semantic space — best overall balance across all metrics.

---

## Conclusion
LDA captures probabilistic word structure but lacks semantic depth. BERTopic captures contextual meaning but offers limited interpretability. This work bridges the gap by converting both models' outputs directly into graph components — probability matrices as edges, word embeddings as nodes — and applying DMoN-based clustering to extract topics that benefit from both.

Applied to 1,015 Korean news articles on AI-based English education, the framework identified 120 latent topics, outperforming both baselines across all three evaluation metrics (C<sub>v</sub>, C<sub>w2v</sub>, Modularity).

<!-- ## Installation

```bash
git clone https://github.com/<your-username>/Topic-GNN-Integration.git
cd Topic-GNN-Integration
pip install -r requirements.txt -->
