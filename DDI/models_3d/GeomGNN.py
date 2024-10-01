import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from layers import SchNet


class GeomGNN(nn.Module):
    """
    This the main class for CIGIN model
    """
    def __init__(self, hidden_dim, num_interactions, cutoff, only_solute = False, no_node_feature = False):
        super(GeomGNN, self).__init__()

        self.geometric_encoder = SchNet(hidden_channels = hidden_dim, num_interactions = num_interactions, cutoff = cutoff)

        self.only_solute = only_solute
        self.no_node_feature = no_node_feature
        
    def forward(self, system, test = False):

        if self.no_node_feature:
            if hasattr(system, "solute"):
                system.z = system.solute.long()
            else:
                system.z = torch.zeros_like(system.z).long()

        geometric_features = self.geometric_encoder(system.z, system.pos, system.batch)
        
        if self.only_solute:
            geometric_features = global_mean_pool(geometric_features[system.solute], system.batch[system.solute])
        else:
            geometric_features = global_mean_pool(geometric_features, system.batch)
        
        return geometric_features