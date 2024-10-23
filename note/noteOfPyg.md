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

`forward`方法
```python
def forward(self,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    ) -> Tensor:
    raise NotImplementedError
```
Compute the score of a **Tuple(head, rel, tail)**

`loss`方法
```python
def loss(self,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor
    ) -> Tensor:
    raise NotImplementedError
```
Compute the loss of a **Tuple(head, rel, tail)**
There is no `loss` function in the most of Pytorch modle, but the loss functions in kge are usually depends on the model. So there is a loss function.

`loader`方法
```python
def loader(self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        **kwgs
        ) -> Tensor:
    '''Return a mini-batch loader that samples a subset of triplets.
    
    Args:
        head_index (torch.Tensor): The head indices.
        rel_type (torch.Tensor): The relation type.
        tail_index (torch.Tensor): The tail indices.
        **kwgs (optional): Additional arguments of 
            :class: 'torch.utils.data.DataLoader', such as 
            :obj: 'batch_size', :obj: 'shuffle', :obj: 'drop_last'
            or :obj:'num_workers'
    '''
    return KGTripletLoader(head_index, rel_type, tail_index, **kwgs)
```
Return a mini-batch loader that samples a subset of triplets.

`test`方法
```python
def test(self,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    batch_size: int,
    k: int = 10,
    log: bool = True
    ) -> Tuple[float, float, float]:
    arange = range(head_index.numel())
    arange = tqdm(arange) if log else arange

    mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        for i in arange:
        h, r, t = head_index[i], rel_type[i], tail_index[i]

        scores = []
        tail_indices = torch.arange(self.num_nodes, device=t.device)
        for ts in tail_indices.split(batch_size):
            scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))
        rank = int((torch.cat(scores).argsort(
            descending=True) == t).nonzero().view(-1))
        mean_ranks.append(rank)
        reciprocal_ranks.append(1 / (rank + 1))
        hits_at_k.append(rank < k)

    mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
    mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
    hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

    return mean_rank, mrr, hits_at_k

```
Compute three index **Mean Rank**, **MRR**, **Hits@k**

`random_sample`方法
```python
def random_sample(self,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]
```
Generate negative samples by changing the head entity or tail entity.

## nn.kge.transe.py


[^nn.embedding]: nn.Embedding 其实只包含一个矩阵，设计成为`nn.Embedding`作为一个`nn.Module`是因为在`weight`之上添加了一些新特性

