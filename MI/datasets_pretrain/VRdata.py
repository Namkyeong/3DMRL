from torch_geometric.data import Data
from typing import (Optional, Dict, Any, Union, List, Iterable, Tuple,
                    NamedTuple, Callable)
from torch_geometric.typing import OptTensor, NodeType, EdgeType
from torch_geometric.deprecation import deprecated

import copy
from collections.abc import Sequence, Mapping

import torch
import numpy as np
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.utils import subgraph
from torch_geometric.data.storage import (BaseStorage, NodeStorage,
                                          EdgeStorage, GlobalStorage)

class VRData(Data):

    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        if pos is not None:
            self.pos = pos

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key:
            return self.num_solvent_atoms
        elif 'face' in key:
            return self.num_nodes
        # elif 'index' in key or 'face' in key:
        #     return self.num_nodes
        elif 'solute_2d_idx' in key:
            return self.solute_size
        elif 'solvent_2d_idx' in key:
            return self.solvent_size
        elif 'mol_idx' in key:
            return self.num_mol
        else:
            return 0