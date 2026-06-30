<p align="right">
  <a href="./README.md">English</a> | <a href="./README_ko.md">한국어</a>
</p>

# 단어 확률 정보와 임베딩 정보를 결합한 그래프 신경망 기반 토픽 모델

> 🎓 **석사학위논문 (2025.12)**
>
> - **저자:** 김순우
> - **지도교수:** [김수현](https://scholar.google.com/citations?hl=ko&user=_qHpoOYAAAAJ)
> - **논문:** https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=5196fd2e578fffceffe0bdc3ef48d419

---

## 이게 뭔가요? 🤔

본 프로젝트는 **확률 기반 토픽 모델링 정보**와 **임베딩 기반 의미 정보**를 하나의 그래프 구조로 통합한  
**그래프 기반 토픽 모델링 프레임워크**입니다.

구체적으로 다음 세 가지 정보를 결합합니다.

- **LDA topic-word probability** → 그래프의 엣지 정보
- **BERTopic/SBERT word embeddings** → 그래프의 노드 특성
- **GCN + DMoN clustering** → 그래프 기반 토픽 클러스터 추출

이 프로젝트의 목표는 단순히 문서를 여러 토픽으로 나누는 것이 아니라,  
**의미적으로 일관되고 구조적으로 해석 가능한 토픽**을 생성하는 것입니다.

---

## 주요 기여점들 💡

- 서로 다른 토픽 모델링 결과를 하나의 **통합 키워드 그래프**로 변환
- LDA의 **확률 기반 단어 관계**와 BERTopic/SBERT의 **임베딩 기반 의미 정보**를 결합
- 비지도 그래프 신경망 클러스터링을 활용한 토픽 발견
- Coherence와 Modularity 지표를 활용하여 토픽 품질 평가

## 기술 스택 🛠

| 구분            | 기술 / 라이브러리                                  | 활용 목적                                            |
| ------------- | ------------------------------------------- | ------------------------------------------------ |
| 개발 환경         | Python, Jupyter Notebook                    | 연구 구현 및 실험 관리                                    |
| 데이터 처리        | Pandas, NumPy, Regular Expressions          | 키워드 전처리, 빈도 필터링, 행렬 데이터 구성                       |
| 토픽 모델링        | LDA, BERTopic                               | 확률 기반 토픽-단어 분포와 의미 기반 토픽 표현 추출                   |
| 임베딩 및 표현 학습   | SBERT / SentenceTransformer                 | 단어 단위 의미 임베딩 생성 및 노드 feature 구성                  |
| 차원 축소 및 클러스터링 | UMAP, HDBSCAN                               | BERTopic 기반 문서 클러스터링 및 토픽 추출                     |
| 그래프 구성        | NetworkX, cosine similarity                 | 토픽-단어 확률과 임베딩 유사도를 활용한 키워드 그래프 생성                |
| 그래프 신경망       | PyTorch, PyTorch Geometric, GCNConv         | 키워드 그래프에 대한 GCN 기반 메시지 패싱 구현                     |
| 그래프 클러스터링     | Custom DMoN-style Clustering, PyTorch       | modularity 기반 비지도 그래프 클러스터링 목적함수를 PyTorch로 직접 구현 |
| 평가            | Gensim CoherenceModel, Word2Vec, Modularity | 토픽 의미 일관성과 그래프 구조 해석 가능성 평가                      |
| 실험 관리         | scikit-learn, tqdm, JSON, Pickle            | feature scaling, 유사도 계산, 진행률 확인, 실험 결과 저장        |


---

## Motivation

본 프로젝트는 

> **서로 다른 성격을 가진 데이터를 하나로 취합하여 잘 활용할 수 있을까?**


에서 출발합니다.

특정 목적을 위해 구성된 데이터는 그 목적을 해결하는 데 적합한 고유한 특성을 가지고 있습니다.  
본 연구는 이러한 특성이 서로 다른 데이터 간에도 효과적으로 결합되고 활용될 수 있는지에 주목했습니다.

이러한 질문을 토픽 모델이라는 환경에서 이질적인 정보 간의 관계를 그래프 구조로 재정의하고,  
각 모델의 결과를 그래프 구조로 변환하여 하나의 프레임워크로 구현했습니다.

---

## Overview

본 저장소는 **LDA**, **BERTopic**, 그리고 **Deep Modularity Networks (DMoN)** 를 결합한  
그래프 기반 토픽 모델링 프레임워크의 전체 구현을 포함합니다.

각 모델의 특징과 한계는 다음과 같습니다.

- **LDA**는 단어의 확률적 동시 등장 구조를 잘 포착하지만, 단어 간 의미적 유사성을 충분히 반영하기 어렵습니다.
- **BERTopic**은 문맥 기반 의미 정보를 잘 반영하지만, 토픽-단어 관계의 해석 가능성에는 한계가 있습니다.
- **제안 방법**은 LDA의 확률 정보와 BERTopic/SBERT의 임베딩 정보를 키워드 그래프에 통합하고, DMoN 클러스터링을 적용하여 구조적·의미적으로 일관된 토픽을 추출합니다.

---

## Methodology

### Stage 1 — Topic Model Output Extraction

먼저 LDA와 BERTopic으로부터 서로 다른 형태의 토픽 정보를 추출합니다.

- **LDA**
  - Topic-word probability matrix 생성
  - 각 단어가 특정 토픽에 속할 확률 정보를 추출

- **BERTopic / SBERT**
  - `paraphrase-multilingual-mpnet-base-v2` 기반 단어 임베딩 생성
  - 각 단어를 384차원 의미 벡터로 표현

---

### Stage 2 — Keyword Graph Construction

| Edge Connections | Node Feature |
| :---: | :---: |
| ![Edge Connections](figures/2.png) | ![Node Feature](figures/3.png) |

키워드 그래프는 다음 두 종류의 엣지로 구성됩니다.

- **Strong edges**
  - 각 LDA 토픽의 Top-K 키워드 사이에 조밀한 intra-topic connection을 생성
  - 동일 토픽 내 핵심 단어들을 강하게 연결
  - Edge weight = 1

- **Weak edges**
  - 핵심 단어가 아닌 non-core word를 가장 유사한 core word와 연결
  - Cosine similarity를 기반으로 연결 강도 부여
  - Edge weight는 0과 1 사이의 값

이를 통해 LDA의 확률 기반 토픽 구조와 SBERT의 의미 기반 단어 표현을  
하나의 키워드 그래프 안에 통합합니다.

---

### Stage 3 — DMoN-based Clustering

![Overall Process](figures/4.png)

- Node features = SBERT word embeddings, 384차원
- 3-layer GCN encoder를 사용하여 keyword graph에서 message passing 수행
- DMoN의 modularity loss와 collapse regularization을 활용하여 구조적으로 안정적인 토픽 클러스터 추출

---

## Data

### Data Collection

본 연구에서는 한국언론진흥재단에서 운영하는 대규모 뉴스 데이터베이스인  
[BIG KINDS](https://www.bigkinds.or.kr)를 통해 뉴스 기사를 수집했습니다.

| Item | Detail |
|------|--------|
| Query | `"생성형 AI 영어교육"` |
| Period | 2023년 1월 ~ 2025년 2월, 총 23개월 |
| # of Articles | 1,015 |
| # of Media Outlets | 88 |
| Variables Used | 기사 본문에서 추출된 키워드 |

---

### Data Preprocessing

추출된 키워드 어휘에 대해 다음 전처리 과정을 수행했습니다.

1. **고유명사 제거**
   - 인명, 기관명, 브랜드명 등 3,100개의 고유명사 제거

2. **빈도 기반 필터링**
   - Zipf's Law를 참고하여 너무 적게 등장하거나 지나치게 자주 등장하는 단어 제거
   - 10회 미만 또는 1,000회 초과 등장 단어 제외

3. **언어 필터링**
   - 한글 단어만 유지
   - 영어, 한자, 숫자, 특수문자 제거

| Stage | Vocabulary Size |
|-------|----------------|
| Raw unique words | 27,182 |
| After preprocessing | **2,597** |

---

## Key Results

제안 방법은 기존 LDA 및 BERTopic 대비 전반적으로 더 높은 토픽 품질을 보였습니다.

| Model | C<sub>w2v</sub> | C<sub>v</sub> | Modularity |
|---|---:|---:|---:|
| LDA | 0.5846 | 0.4391 | 0.1910 |
| BERTopic | 0.6779 | 0.4754 | 0.2347 |
| **Ours (DMoN)** | **0.7220** | **0.5621** | **0.3454** |

![Performance Comparison](figures/5.png)

---

## Ablation Study

### Effect of Top-K Setting

Top-K는 각 LDA 토픽에서 몇 개의 핵심 단어를 선택할 것인지를 의미하며,  
그래프의 밀도와 클러스터 구조에 큰 영향을 줍니다.

본 실험에서는 토픽 수를 30으로 고정하고, Top-K 값을 10부터 50까지 변화시키며 비교했습니다.

| Top-K | Total Edges | Strong | Weak | Modularity | C<sub>w2v</sub> | C<sub>v</sub> |
|-------|-------------|--------|------|------------|--------|-----|
| 10 | 3,198 | 1,145 | 2,053 | 0.5843 | 0.5405 | 0.7089 |
| 20 | 6,529 | 4,561 | 1,968 | 0.4482 | 0.5575 | 0.7160 |
| **✅ 30**<br><sub>Best balance</sub> | **11,960** | **10,075** | **1,885** | **0.3454** | **0.5621** | **0.7220** |
| 40 | 19,113 | 17,300 | 1,813 | 0.2823 | 0.5595 | 0.7268 |
| 50 | 27,916 | 26,169 | 1,747 | 0.1866 | 0.5500 | 0.7218 |

Top-K가 증가할수록 strong edge의 비중이 커지며,  
노드 간 연결이 과도하게 조밀해지는 경향이 나타났습니다.

그 결과 클러스터 간 경계가 흐려지고 modularity가 감소했습니다.  
실험 결과 **Top-K = 30**이 semantic coherence와 graph modularity 사이에서 가장 균형 잡힌 설정으로 나타났습니다.

---

### Effect of Edge Configuration

Strong edge와 weak edge의 역할을 확인하기 위해 세 가지 그래프 구성 방식을 비교했습니다.

| Configuration | Total Edges | Strong | Weak | Modularity | C<sub>w2v</sub> | C<sub>v</sub> |
|---------------|-------------|--------|------|------------|--------|-----|
| Weak-only | 1,885 | 0 | 1,885 | 0.6437 | 0.5271 | 0.6992 |
| **✅ Strong+Weak**<br><sub>Proposed</sub> | **11,960** | **10,075** | **1,885** | **0.3454** | **0.5621** | **0.7220** |
| Strong-only | 10,075 | 10,075 | 0 | 0.2960 | 0.5696 | 0.7200 |

- **Weak-only**
  - 가장 높은 modularity를 보이며 클러스터 경계가 명확하게 나타남
  - 그러나 주변 단어들이 의미적으로 충분히 묶이지 않아 semantic coherence가 낮게 나타남

- **Strong-only**
  - C<sub>v</sub>와 C<sub>w2v</sub>는 비교적 높게 나타남
  - 그러나 edge density가 높아져 클러스터 경계가 흐려지는 문제가 발생

- **Strong+Weak (Proposed)**
  - Strong edge가 각 토픽의 의미적 중심을 형성
  - Weak edge가 주변 단어를 연결하여 의미 공간을 확장
  - 세 지표를 종합했을 때 가장 균형 잡힌 결과를 보임

---

## Conclusion

LDA는 단어의 확률적 구조를 잘 포착하지만 의미적 깊이가 부족하고,  
BERTopic은 문맥 기반 의미 표현에 강점이 있지만 토픽-단어 관계의 해석 가능성에는 한계가 있습니다.

본 연구는 두 모델의 출력을 그래프 구성 요소로 변환했습니다.

- LDA topic-word probability → graph edge
- SBERT word embedding → node feature

이후 DMoN 기반 그래프 클러스터링을 적용하여  
확률적 구조와 의미적 정보를 함께 반영한 토픽을 추출했습니다.

AI 기반 영어교육 관련 한국어 뉴스 기사 1,015건에 적용한 결과,  
제안 프레임워크는 120개의 잠재 토픽을 도출했으며,  
LDA와 BERTopic 대비 C<sub>v</sub>, C<sub>w2v</sub>, Modularity 지표에서 모두 더 나은 성능을 보였습니다.

---

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

