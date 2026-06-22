# Graph Neural Network-Based Topic Model Integrating Probabilistic and Embedding Information of Words

> 🎓 **Master's Thesis (Dec 2025)**
> - **Author:** Soonwoo Kim 
> - **Supervisor:** Prof. Suhyeon Kim
> - **Link:** https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=5196fd2e578fffceffe0bdc3ef48d419


## What is this? 🤔

This project proposes a graph-based topic modeling framework that combines:

- LDA topic-word probability as graph edges
- BERTopic/SBERT word embeddings as node features
- GCN + DMoN clustering for topic extraction

The goal is to generate topics that are both semantically coherent and structurally interpretable.

## Key Contributions 💡

- Converted heterogeneous topic modeling outputs into a unified keyword graph
- Integrated probabilistic and embedding-based word information
- Applied unsupervised graph neural network clustering for topic discovery
- Evaluated topic quality using coherence and modularity metrics

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
| **✅ 30**<br><sub>Best balance</sub> | **11,960** | **10,075** | **1,885** | **0.3454** | **0.5621** | **0.7220** |
| 40 | 19,113 | 17,300 | 1,813 | 0.2823 | 0.5595 | 0.7268 |
| 50 | 27,916 | 26,169 | 1,747 | 0.1866 | 0.5500 | 0.7218 |

As K increases, strong edges dominate (35.8% → 93.7%), homogenizing node connectivity and blurring cluster boundaries — modularity drops ~68%. **K=30** achieves the best balance across all three metrics.

---

### Effect of Edge Configuration

Three edge configurations were tested to verify the contribution of the strong+weak design.

| Configuration | Total Edges | Strong | Weak | Modularity | C<sub>w2v</sub> | C<sub>v</sub> |
|---------------|-------------|--------|------|------------|--------|-----|
| Weak-only | 1,885 | 0 | 1,885 | 0.6437 | 0.5271 | 0.6992 |
| **✅ Strong+Weak**<br><sub>Proposed</sub> | **11,960** | **10,075** | **1,885** | **0.3454** | **0.5621** | **0.7220** |
| Strong-only | 10,075 | 10,075 | 0 | 0.2960 | 0.5696 | 0.7200 |

- **Weak-only**: Highest modularity (clearest cluster boundaries) but lowest semantic coherence — peripheral words lack sufficient semantic binding.
- **Strong-only**: Higher C<sub>v</sub> and C<sub>w2v</sub> but blurred cluster boundaries due to excessive edge density.
- **Strong+Weak (Proposed)**: Strong edges anchor the semantic core of each topic; weak edges link peripheral words and expand the semantic space — best overall balance across all metrics.

---

## Conclusion
LDA captures probabilistic word structure but lacks semantic depth. BERTopic captures contextual meaning but offers limited interpretability. This work bridges the gap by converting both models' outputs directly into graph components — probability matrices as edges, word embeddings as nodes — and applying DMoN-based clustering to extract topics that benefit from both.

Applied to 1,015 Korean news articles on AI-based English education, the framework identified 120 latent topics, outperforming both baselines across all three evaluation metrics (C<sub>v</sub>, C<sub>w2v</sub>, Modularity).

## References
[1] R. Kune, P. K. Konugurthi, A. Agarwal, R. R. Chillarige, and R. Buyya, “The Anatomy of Big Data Computing,” Journal of Computer and Information Sciences, pp. 1–12, 2015.

[2] M. Shah, “Big Data and the Internet of Things,” Research and Technology Center – North America, Robert Bosch LLC, Palo Alto, USA, 2015. arXiv: 1503.07092v1

[3] U. Sivarajah, M. M. Kamal, Z. Irani, and V. Weerakkody, “Critical Analysis of Big Data Challenges and Analytical Methods,” Journal of Business Research, vol. 70, pp. 263–286, 2017.

[4] C. Dobre and F. Xhafa, “Intelligent Services for Big Data Science,” Future Generation Computer Systems, vol. 37, pp. 267–281, 2014.

[5] M. Sadia, A. R. Chowdhury, and A. Chen, “A Case for Computing on Unstructured Data,” arXiv preprint arXiv:2509.14601, 2025.

[6] M. Gentzkow, B. Kelly, and M. Taddy, “Text as Data,” Journal of Economic Literature, vol. 57, no. 3, pp. 535–574, 2019.

[7] E. Landhuis, “Information overload: How to manage the research-paper deluge?,” Nature, vol. 535, pp. 457–458, Jul. 2016.

[8] O. Azeroual, G. Saake, M. Abuosba, and J. Schöpfel, “Text Data Mining and Data Quality Management for Research Information Systems in the Context of Open Data and Open Science,” Proceedings of the International Conference on Open Access (ICOA 2018), pp. 1–17, 2018.

[9] H. Hassani, C. Beneki, S. Unger, M. T. Mazinani, and M. R. Yeganegi, “Text Mining in Big Data Analytics,” Big Data and Cognitive Computing, vol. 4, no. 1, p. 1, 2020.

[10] D. M. Blei, “Probabilistic Topic Models,” Communications of the ACM, vol. 55, no. 4, pp. 77–84, Apr. 2012.

[11] D. M. Blei and J. D. Lafferty, “Topic Models,” in Text Mining: Classification, Clustering, and Applications, A. N. Srivastava and M. Sahami (eds.), Chapman and Hall CRC, pp. 71–94, 2009.

[12] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent Dirichlet Allocation,” Journal of Machine Learning Research, vol. 3, pp. 993–1022, 2003.

[13] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention Is All You Need,” Advances in Neural Information Processing Systems (NeurIPS 2017), pp. 5998–6008, 2017. arXiv:1706.03762

[14] Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT), pp. 4171–4186, 2019. arXiv:1810.04805

[15] M. Grootendorst, “BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure,”, 2022. arXiv preprint arXiv:2203.05794

[16] A. Tsitsulin, J. Palowitch, B. Perozzi, and E. Müller, “Graph Clustering with Graph Neural Networks,” Journal of Machine Learning Research, vol. 24, pp. 1–21, 2023.

[17] G. Salton and C. Buckley, “Term-weighting approaches in automatic text retrieval,” Information Processing & Management, vol. 24, no. 5, pp. 513–523, 1988.

[18] T. Joachims, “A probabilistic analysis of the Rocchio algorithm with TFIDF for text categorization,” Proceedings of the International Conference on Machine Learning (ICML), pp. 143–151, 1997.

[19] L. McInnes, J. Healy, and J.Melville, “UMAP: Uniform manifold approximation and projection for dimension reduction,” arXiv preprint arXiv:1802.03426, 2020.

[20] L. McInnes, J. Healy, and S. Astels, “hdbscan: Hierarchical density based clustering,” Journal of Open Source Software, vol. 2, no. 11, p. 205, 2017.

[21] T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” Proceedings of the International Conference on Learning Representations (ICLR), pp. 1–14, 2017. arXiv preprint arXiv:1609.02907
[22] BIG Kinds: https://www.bigkinds.or.kr

[23] G. K. Zipf, Human Behavior and the Principle of Least Effort: An Introduction to Human Ecology, Addison-Wesley Press, Cambridge, MA, 1949.

[24] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient Estimation of Word Representations in Vector Space”, Sep. 2013. arXiv preprint arXiv:1301.3781

[25] D. O’Callaghan, D. Greene, J. Carthy, and P. Cunningham, “An analysis of the coherence of descriptors in topic modeling”, Expert Systems with Applications, vol. 42, no. 13, pp. 5645–5657, 2015.

[26] G. Bouma, “Normalized (Pointwise) Mutual Information in Collocation Extraction,” in From Form to Meaning: Processing Texts Automatically — Proceedings of the Biennial GSCL Conference 2009, Tübingen: Gunter Narr Verlag, pp. 31–40, 2009.

[27] M. E. J. Newman, “Modularity and community structure in networks”, Proceedings of the National Academy of Sciences of the United States of America, vol. 103, no. 23, pp. 8577–8582, Jun. 2006.

[28] Seoul Metropolitan Government, “A pilot service of Metaverse Seoul is launched,” Press Release, Seoul, South Korea, May 10, 2022. [Online]. Available: https://english.seoul.go.kr/a-pilot-service-of-metaverse-seoul-is-launched/

[29] World Bank, “Teachers are leading an AI revolution in Korean classrooms,” World Bank Blogs, Oct. 30, 2024. [Online]. Available: https://blogs.worldbank.org/en/education/teachers-are-leading-an-ai-revolution-in-korean-classrooms

[30] Ministry of Education, Republic of Korea, “AI Digital Textbooks for 2025 to Realize Personalized Learning,” Policy Document, Nov. 29, 2024. [Online]. Available: https://english.moe.go.kr/boardCnts/viewRenewal.do?boardID=265&boardSeq=102075&lev=0&searchType=null&statusYN=W&page=1&s=english&m=0201&opType=N


<!-- ## Installation

```bash
git clone https://github.com/<your-username>/Topic-GNN-Integration.git
cd Topic-GNN-Integration
pip install -r requirements.txt -->
