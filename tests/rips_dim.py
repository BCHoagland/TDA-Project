# tests to see if changing max_dimension of the rips complex affects the 0-dimensional death times

import torch
import torch.nn as nn
import gudhi as gd

class Rips(nn.Module):
    def __init__(self, max_edge_length, max_dimension):
        super(Rips, self).__init__()
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension

    def forward(self, input):
        rips = gd.RipsComplex(points=input, max_edge_length=self.max_edge_length)
        st = rips.create_simplex_tree(max_dimension=self.max_dimension)
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


l = 5
rips0 = Rips(l, 0)
rips1 = Rips(l, 1)
rips2 = Rips(l, 2)

N = 1000
pts = torch.randn(N, 20)        # N normally distributed 5-dimensional points
dgm0 = rips0(pts)
dgm1 = rips1(pts)
dgm2 = rips2(pts)

# print(dgm0)
# print(dgm1)
# print(dgm2)

print(not False in dgm0 == dgm1)
print(not False in dgm1 == dgm2)
