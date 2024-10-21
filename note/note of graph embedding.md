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
* 平移距离模型主要运用基于距离的评分函数来对关系进行打分，以此来进行训练（TransE，TransH，TransR）
* 语义匹配模型利用基于相似性的评分函数。主要通过匹配实体的潜在语义和向量空间表示中包含的关系来度量事实的可信度

# Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding
Introduce **OrthogonalE** employing matrices for entities and block-diagonal orthogonal matrices with Rie-mannian optimization for relations

# RotatE
Use **Hadamard** product to multiply the real and imaginary components of the head entity embedding by the **angle-based relation embedding**.[^1]resulting in a **2D rotation effect** within each unit. Each unit consists of two elements representing the real and imaginary components.[^2]For example, an entity embedding with 500 dimensions consists of 250 units. The resulting vectors in each unit are then concatenated to form the tail entity embedding.
#

[^1]: **Hadamard product**？**head entity embedding**？**real and imaginary components**？**angle-based relation embedding**？
[^2]: **2D rotation effect**?
