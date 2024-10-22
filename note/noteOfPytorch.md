# nn
## nn.Embedding
```python
class torch.nn.Embedding(num_embeddings,
 embedding_dim, 
 padding_idx=None, 
 max_norm=None, 
 norm_type=2.0, 
 scale_grad_by_freq=False, 
 sparse=False, 
 _weight=None, 
 _freeze=False, 
 device=None, 
 dtype=None)
```
### Parameters
* `padding_idx(int, optional)` if specified, the entries at `padding_idx` do not contribute to the gradient. therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”. For a newly constructed Embedding, the embedding vector at padding_idx will default to all zeros, but can be updated to another value to be used as the padding vector.
位于`padding_idx`位置的embedding不参与更新，并且在新建Embedding时，位于`padding_idx`的向量会背设置为零向量
* `max_norm(float, optional)` if given, each embedding vector with norm larger than `max_norm` is renormalized to have norm `max_norm`
这个参数会限制embedding vector的norm
* `norm_type(float, optional)` The `p` of the p-norm to compute the `max_norm` option. Default `2`
* `scale_grad_by_freq(bool, optional)` If given, this will scale gradient by the inverse of frequency of the words in the mini-batch. Default `False`
* `sparse(bool, optional)` If `True`, gradient w.r.t `weight` matrix will be a sparse tensor.
### Variables
**weight(Tensot)** -the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from $N(0, 1)$
* Input(*), IntTensor or LongTensor of arbitrary shape containing the indices to extract
* Output(*, H), where * is the input shape and $H = embedding_dim$

**Note**
when `max_norm` is not `None`, `Embedding's` forward method will modify the `weight` tensor in-place. Since tensors needed for gradient computations can not be modified in-place, performing a differentiable operation on `Embedding.weight` before calling `Embedding's` forward mehod requires cloning `Embedding.weight` when `max_norm` is not `None`
如果在修改之前试图对`embedding.weight`执行某些可以微分的操作(`@`)，那么无法为这些操作的参数计算梯度。所以要想使`embedding.weight`参与某些可微分计算，并且正确地得到梯度，必须在操作之后，且在`backward`之前不能使用`forward()`。如果使用`forward`会对本来的`embedding.weight`进行修改，从而导致pytorch追踪不到最初的weight。
