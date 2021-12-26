import torch
import torch.nn as nn
import gudhi as gd


class Rips(nn.Module):
    def __init__(self, max_edge_length):
        super(Rips, self).__init__()
        self.max_edge_length = max_edge_length

    def forward(self, input):
        rips = gd.RipsComplex(points=input, max_edge_length=self.max_edge_length)
        st = rips.create_simplex_tree(max_dimension=0)
        st.compute_persistence()
        idx = st.flag_persistence_generators()

        # 1-D
        # if len(idx[1]) == 0:
        #     verts = torch.empty((0, 4), dtype=int)
        # else:
        #     verts = torch.tensor(idx[1][0])
        # dgm = torch.norm(input[verts[:, (0, 2)]] - input[verts[:, (1, 3)]], dim=-1)

        # 0-D
        if len(idx[0]) == 0:
            verts = torch.empty((0, 2), dtype=int)
        else:
            verts = torch.tensor(idx[0][:, 1:])
        dgm = torch.norm(input[verts[:,0], :] - input[verts[:,1], :], dim=-1)

        return dgm
