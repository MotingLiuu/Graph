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


# RotatE
Use **Hadamard** product to multiply the real and imaginary components of the head entity embedding by the **angle-based relation embedding**.[^1]resulting in a **2D rotation effect** within each unit. Each unit consists of two elements representing the real and imaginary components.[^2]For example, an entity embedding with 500 dimensions consists of 250 units. The resulting vectors in each unit are then concatenated to form the tail entity embedding.
#

[^1]: **Hadamard product**？**head entity embedding**？**real and imaginary components**？**angle-based relation embedding**？
[^2]: **2D rotation effect**?
[^translation-based]: **TransE**(Translating Embeddings for Modeling Multi-relational Data) **TransH**(Knowledge Graph Embedding by Translating on Hyperplanes) **TransR**(Learning Entity and Relation Embedding for Knowledge Graph)
[^ideaFromWord2vec]: The idea of **TransE** is from **Word2vec**. 完全照抄
[^initializeE]: sample from a uniform distribution $e \sim uniform(-\frac{6}{\sqrt{k}}, \frac{6}{\sqrt{k}})$ 每个实体和关系会被表示成一个从均匀分布中抽取的k维向量
[^lossFunction]: This method, sampling some negtive samples to get the loss, is similar to **negtive sampleing** in word2vec. 论文中提到要将embedding的norm限制为1，否则训练的时候会让负样本相关的norm变大，从而导致损失函数下降。对比word2vec论文中使用negativeSampling的损失函数$Loss_{neg}(v_c,o,U)=-log(\sigma(\mu_o^Tv_c))-\sum_{k=1}^K {log(\sigma(-\mu_k^Tv_c))}$,发现这里使用了sigmoid这个函数，不容易计算导数，而且可以将内积映射到-1~1的值域上，从而使得norm对于loss没有很大的影响（两个向量保持夹角相同，调整向量的norm）。