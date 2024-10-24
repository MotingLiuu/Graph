# Knowledge Graph Embedding(KGE)
**Learn to low-dimensional representations of entities and relattions for predicting missing facts.**
Rotation-based methods like **RotatE** and **QuatE** have two challenges:

1. limited model flexibility requiring proportional increases in relation size with entity dimension
2. difficulties in generalizing the model for higher-dimension rotations.

知识图谱是由实体和关系组成的一个网络，实体对应网络中的节点，关系为网络中的边，边表示为（头实体，关系，尾实体）的三元组。
知识图谱存储形式：
* 三元组（A，RED，D）
* 用不同关系区分成不同矩阵进行存储，关系A生成一个邻接矩阵，关系B生成一个邻接矩阵，以此类推

KGE主要分为两大类：平移距离模型和语义匹配模型
* 平移距离模型主要运用基于距离的评分函数来对关系进行打分，以此来进行训练（TransE，TransH，TransR）[^translation-based]
* 语义匹配模型利用基于相似性的评分函数。主要通过匹配实体的潜在语义和向量空间表示中包含的关系来度量事实的可信度

# Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding
Introduce **OrthogonalE** employing matrices for entities and block-diagonal orthogonal matrices with Rie-mannian optimization for relations

# TransE
Models relationships by interpreting them as **translations operating** on low-dimensional embeddings of entities.
Energy-based model for learning low-dimensional embeddings of entities. 
In **TransE**, relationships are rpresented as ***translations*** in the embedding of space. if $(h,l,t)$ holds, then the embedding of the tail entity ***t*** should be close to the embedding of head entity ***h*** plus some vector that depends on the relationship ***l***.[^ideaFromWord2vec]

To Learn sunch embedding, minimize a margin-based ranking criterion over the training set:[^lossFunction]
$$
L = \sum_{(h, l, t) \in S}\sum_{(h',l',t')\in S'_{(h,l,t)}} [\gamma + d(h+l,t) - d(h'+l,t')]_+
$$
$\gamma > 0$ is a margin hyperparameter, and
$$
S'_{(h,l,t)} = \{(h',l,t)|h' \in E\}\cup \{(h,l,t')|t'\in E\}
$$
### Algorithm 1: Learning TransE

**Input**: Training set \( S = \{(h, l, t)\} \), entities and relation sets \( E \) and \( L \), margin \(\gamma\), embedding dimension \(k\).

1. **Initialization**:
   - Initialize relation embeddings \( r \sim \text{uniform}\left(-\frac{6}{\sqrt{k}}, \frac{6}{\sqrt{k}}\right) \) for each relation \( r \in L \).
   - Initialize entity embeddings \( e \sim \text{uniform}\left(-\frac{6}{\sqrt{k}}, \frac{6}{\sqrt{k}}\right) \) for each entity \( e \in E \).[^initializeE]

2. **Training Loop**:
   - Repeat until convergence:
     1. Normalize embeddings: \( e = \frac{e}{||e||} \) for each entity \( e \in E \).
     2. Sample a mini-batch \( S_{\text{batch}} \) of size \( b \) from the training set.
     3. Initialize an empty set \( T_{\text{batch}} \) for pairs of triplets.
     4. For each positive triplet \( (h, l, t) \in S_{\text{batch}} \):
        - Sample a corrupted triplet \( (h', l, t') \sim S(h, l, t) \).
        - Add the pair \( ((h, l, t), (h', l, t')) \) to \( T_{\text{batch}} \).
     5. Update embeddings with respect to the following objective:
        $$ 
        \sum_{(h, t), (h', t') \in T_{\text{batch}}} \nabla\left[ \gamma + d(h + l, t) - d(h' + l, t') \right]_+ 
        $$
        where \( [x]_+ = \max(0, x) \) and \( d(\cdot) \) represents the distance function.

## Comment
* 这篇文章的思路和word2vec完全相同，而且由于KG向量+关系可以自然而然地推断出另外一个向量，相对于bagOfWord更加直观。完全是word2vec思想的利用。
* 没有像word2vec一样建立一个概率模型，而是简单粗暴地将embedding建模为k维向量（norm为1），接着优化embedding向量，减小loss。
* loss的设计借鉴negativesampling，但word2vec中使用了sigmoid将向量的内积映射到了0~1值域上，使得norm对于loss的影响没有特别巨大。但是这篇论文中并没有相关的设计，只是单纯地限制了norm。所以如果用上sigmoid函数，或者说利用范数无关的损失函数来进行训练会如何？


* The approach in this paper is exactly the same as that of word2vec, and since KG embeddings + relations can naturally infer another vector, it is more intuitive compared to bag-of-words. It is essentially a direct application of word2vec's idea.
* Unlike word2vec, which builds a probabilistic model, this paper takes a more straightforward approach by modeling the embeddings as k-dimensional vectors (normalized to have unit norm), then optimizing the embedding vectors to minimize the loss.
* The loss function design draws on negative sampling, but in word2vec, a sigmoid function is used to map the inner product of vectors to the range of 0 to 1, thus reducing the impact of the norm on the loss. However, this paper does not employ such a mechanism and only restricts the norm. So, how would the results change if we introduced a sigmoid function or trained the model with a norm-independent loss function?


# RotatE
Use **Hadamard** product to multiply the real and imaginary components of the head entity embedding by the **angle-based relation embedding**.[^1]resulting in a **2D rotation effect** within each unit. Each unit consists of two elements representing the real and imaginary components.[^2]For example, an entity embedding with 500 dimensions consists of 250 units. The resulting vectors in each unit are then concatenated to form the tail entity embedding.
#

[^1]: **Hadamard product**？**head entity embedding**？**real and imaginary components**？**angle-based relation embedding**？
[^2]: **2D rotation effect**?
[^translation-based]: **TransE**(Translating Embeddings for Modeling Multi-relational Data) **TransH**(Knowledge Graph Embedding by Translating on Hyperplanes) **TransR**(Learning Entity and Relation Embedding for Knowledge Graph)
[^ideaFromWord2vec]: The idea of **TransE** is from **Word2vec**. 完全照抄
[^initializeE]: sample from a uniform distribution $e \sim uniform(-\frac{6}{\sqrt{k}}, \frac{6}{\sqrt{k}})$ 每个实体和关系会被表示成一个从均匀分布中抽取的k维向量
[^lossFunction]: This method, sampling some negtive samples to get the loss, is similar to **negtive sampleing** in word2vec. 论文中提到要将embedding的norm限制为1，否则训练的时候会让负样本相关的norm变大，从而导致损失函数下降。对比word2vec论文中使用negativeSampling的损失函数$Loss_{neg}(v_c,o,U)=-log(\sigma(\mu_o^Tv_c))-\sum_{k=1}^K {log(\sigma(-\mu_k^Tv_c))}$,发现这里使用了sigmoid这个函数，不容易计算导数，而且可以将内积映射到-1~1的值域上，从而使得norm对于loss没有很大的影响（两个向量保持夹角相同，调整向量的norm）。

# Knowledge Graph Embedding Compression
## Abstract
Representing each entity in the KG as **a vector of discrete codes** and then composes the embeddings from these codes. Replace the traditional KG embedding layer by representing each entity in the KG with a K-way D dimensional code(**KD code**).
用离散编码组成的vector（KD code）？
**Discretization** and **reverse-discretization** function are leart end-to-end.
两个阶段，一个离散化，一个逆离散化。
Inherent discreteness of the representation learning problem.
解决离散表示学习固有的问题（We tackle these issues by resorting to the straight-through estimator (Bengio
 et al., 2013) or the tempering softmax (Maddison et al., 2016; Jang et al., 2016) and using guidance from existing KG embeddings to smoothly guide learning of the discrete representations.）

 ## Discrete Representation Learning
 **Representing each symbol $v$ in the vocabulary as a discrete vector $Z_v=[Z_v^{(1)},...,Z_v^{(D)}]$
Discrete representation learning approaches (van den Oord et al., 2017; Chen et al.,2018; Chen and Sun, 2019). 

## Discrete KG Representation Learning
Define a **quantization function** $Q:R^d \to R^d$, which takes raw KG embeddings and produces their quantized representation. $Q = D \circ R$ is composed of two functions:
   1. **A discretization function** $D: R^{d_e} \to Z^D$ (vector quantization)
   2. **A reverse-discretization function** $R: Z^D \to R^{d_e}$ (codebook learning)
During **training**, both $D$ and $R$ are learned. Then every entity in the KG is represented by a KD code via applying the discretization function $D$ to save space(compression). 
During **testing/inference** stage, the reverse-discretization function $R$ is used to decode the KD codes into regular embedding vectors.

### Discretization Function $D$
Map continuous **KD embedding** vectors into **KD code**. 
Model the discretization function using nearest neighbor search(Cayton, 2008). Given continuous KG embeddings $\{ e_i|i=1...n_e\}$ as query vectors, we define a set of **K** key vectors $\{K_k|k=1...K\}$ where $k_k^{(i)} \in R^{K\times d_e/D}, j= 1...D$

Partition the query and key vectors into $D$ partitions where each partition corresponds one of the $D$ discrete codes $-e_i^{(j)} \in R^{K\times d_e /D}$ and $k_k^{(j)} \in R^{K\times d_e /D}, j=1...D$

**Vector Quantization(VQ)**
$j^{th}$ discrete code of $i^{th}$ entity $z^{(j)}_i$ can be computed by calculating distances between the corresponding query partition $e_i^{(j)}$ and various corresponding key vector partitions $\{ K_k^{(j)}\}$
$$
Z_i^{(j)} == argmin_k \ dist(e_i^{(j)}, K_k^{(j)})
$$
in the experiment
$$
dist(a, b) = ||a - b||^2_2
$$
use the straight-through estimator to compute a pseudo gradient.(ben-gio et al., 2013)

**Tempering Softmax(Ts)**
continuous relaxation of $Z_i^{(j)} == argmin_k \ dist(e_i^{(j)}, K_k^{(j)})$ using TS(Maddison son et al. 2016; jang et al.,2016).
$$
Z_i^{(j)} == argmax_k \frac{exp(<e_i^{(j)}, k_k^{(j)}>/r)}{\sum_{k'}exp(<e_i^{(j)},k_{k'}^{(j)}>/r)}
$$
