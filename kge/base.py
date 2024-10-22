import torch
from torch import Tensor
from torch.nn import Embedding
from tqdm import tqdm

from typing import Tuple

from torch_geometric.nn.kge.loader import KGTripletLoader

class KGEModel(torch.nn.Module):
    '''An abstract base class for implementing custom KGE models
    
    Args:
        num_nodes (int): The number of nodes/entities in the graph
        num_relations (int): The number of relations in the graph
        hidden_channels (int): The hidden embedding size
        sparse (bool, optional): if set to :obj:'True', gradients w.r.t to the embedding matrices will be sparse. (default: :obj:'False')
    '''
    def __init__(self,
                 num_nodes: int,
                 num_relations: int,
                 hidden_channels: int, 
                 sparse: bool = False
                 ):
        
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels
        
        self.node_emb = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)
        
    def reset_parameter(self):
        '''Reset all learnable parameters of the module'''
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()
        
    def forward(self,
                head_index: Tensor,
                rel_type: Tensor,
                tail_index: Tensor
                ) -> Tensor:
        '''Return the score for the given triplet.
        
        Args:
            head_index (torch.Tensor): The head indices.
            rel_index (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        '''
        raise NotImplementedError
    
    def loss(self,
             head_index: Tensor,
             rel_type: Tensor,
             tail_index: Tensor
             ) -> Tensor:
        '''Returns the loss value for the given triplet.
        
        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        '''
        raise NotImplementedError
    
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