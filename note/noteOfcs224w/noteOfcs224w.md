# Outline
![alt text](image.png)
## Tasks
![alt text](image-1.png)
# Leacture 1

**Node Degree**
![alt text](image2.png)

**Bipartite graph**
![alt text](image3.png)
**Floded/Projected Bipartitile Graphs**
![alt text](image4.png)

**Connectivity**
![alt text](image-2.png)

# Lecture 2 Traditional Feature

## Feature Design
* Node-level prediction
* Link-level prediction
* Graph-level prediction

Focus on **undirected graphs**

**Goal** Characterize the structure and position of a node in the netword:

Two categories: **Importance-based features**(Node degrees, Different node Centrality measures) **Structure-based features**(Node degrees, Clustering coefficient, Graphlet count vector)
* Node degree
* Node centrality: takes the node **importance in a graph** into accout
    * **Eigenvector centrality**
    A node $v$ is important if surrounded by important neighboring nodes $u \in N(v)$.
    $$
    c_v = \frac{1}{\lambda}\sum_{\mu \in N(v)} c_\mu
    $$
    equals to
    $$
    \lambda c = Ac
    $$
    The largest eigenvalue $\lambda_{max}$ is always positive and unique.
    The leading eigenvector $c_{max}$ is used for centrality.
    一旦给出一个邻接矩阵A，就可以确定唯一一个$\lambda$和centrality vector $c$

    * **Betweenness centrality**
    A node is important if it lies on many shortest paths between other nodes.
    $$
    c_v = \sum_{s \neq v \neq t} \frac{\#(shortest\ path\ betwens\ s\ and t\ that\ contain\ v)}{\#(shortest\ paths\ between\ s\ and\ t)}
    $$
    用node在最短通路的数量来衡量node在graph中的重要程度

    * **Closeness centrality**
    $$
    c_v = \frac{1}{\sum_{u \neq v}shortest\ path\ length\ between\ u\ and\ v}
    $$
    用node到其他节点的容易程度来衡量
* Clustering coefficient
    ![alt text](image-3.png)
    **Observation** counts the traiangles(which are constructed by nodes including the current node) in the ego-netword
    ![alt text](image-4.png)
    如图所示，$C_2^4 = 6$所以包含当前节点在内可以构成6个三角形。因为多一个边，就是多了一个三角形的底边（包含当前节点构成三角形）。所以计算周围邻居点之间的边，和计算包含当前节点构成的三角形是一样的。
    衡量node链接区域的整体连通性

* Graphlets
    Rooted connected non-isomorphic
    ![alt text](image-5.png)
    * Degree counts edges that a node touches
    * Clustering coefficient counts triangles that a node touches
    * Graphlet Degree Vector(GDV) counts graphlets that a node touches
    **Graph Degree Vector (GDV)**: A count vector of graphlets rooted at a given node.
    ![alt text](image-6.png)
    rooted的意思是，当前节点作为结构中的根节点。一个结构可能有多个根节点，会产生多个rooted graph如上图中的c结构和d结构。