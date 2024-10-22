# nn.kge

This package is used to make kge model, including `__init__.py`, `base.py`, `complex.py`, `distmult.py`, `loder.py`, `rotate.py`, `transe.py`

## kge.base.py
This file defines `class: :KGEModel:` which is a abstraction of kge models.

```python
class KGEModel(torch.nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, hidden_channels: int, sparse: bool = False)
    def reset_parameter(self)
    def forward(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor
    def loss(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor
    def loader(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor, **kwargs) -> Tensor
    @torch.no_grad()
    def test((self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor, batch_size: int, k: int = 10, log: bool = True) -> Tuple[float, float, float]
    @torch.no_grad()
    def random_sample((self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]
    def __repr__(self) -> str
```

构造函数`__init__`
```python
def __init__(
    self,
    num_nodes: int,
    num_relations: int,
    hidden_channels: int,
    sparse: bool = False,
):
    super().__init__()
        
    self.num_nodes = num_nodes
    self.num_relations = num_relations
    self.hidden_channels = hidden_channels
        
    self.node_emb = Embedding(num_nodes, hidden_channels, sparse=sparse)
    self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)

```
* `sparse`: 若为`True`则用稀疏梯度更新嵌入矩阵
* 这个函数创建了两个`Embedding`矩阵，分别用来存储node和relation的embedding矩阵

`reset_parameters`方法
```python
def reset_parameter(self):
    self.node_emb.reset_parameters()
    self.rel_emb.reset_parameters()
```
* 重置所有可学习的参数[^nn.embedding]


[^nn.embedding]: nn.Embedding
