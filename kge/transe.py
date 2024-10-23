import math
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import KGEModel

class TransE(KGEModel):
    '''TransE model 'Translating Emebdding for Modeling Multi-Relational Data'
    
    Args:
        num_nodes
        num_relations
        hidden_channels
        margin
        p_norm
        sparse
    '''
    
    def __init__(self,
                 num_nodes: int,
                 num_relations: int,
                 hidden_channels: int,
                 margin: float = 1.0,
                 p_norm: float = 1.0,
                 sparse: bool = False
                 ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        
        self.p_norm = p_norm
        self.margin = margin
        
        self.reset_parameters()
        
    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1, out=self.rel_emb.weight.data)
        
    def forward(self,
                head_index: Tensor,
                rel_type: Tensor,
                taril_index: Tensor,
                ) -> Tensor:
        
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(taril_index)
        
        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)
        
        return -((head + rel) - tail).norm(p=self.p_norm, dim = 1)
    
    def loss(self,
             head_index: Tensor,
             rel_type: Tensor,
             tail_index: Tensor
             ) -> Tensor:
        
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        
        return F.margin_ranking_loss(pos_score,
                                     neg_score,
                                     target=torch.ones_like(pos_score),
                                     margin=self.margin
                                     )