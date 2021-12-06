# tests to see where points inside the unit ball when maximizing the sum of
# the 0-dimensional persistent homology death times
# while keeping them inside the unit ball

import torch
import torch.nn as nn
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

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

# get the sum of the death times while tracking gradients
def h_loss(pts, rips):
    deaths = rips(pts)
    total_0pers = torch.sum(deaths)
    # total_0pers = torch.min(deaths)
    disk = (pts ** 2).sum(-1) - 1
    disk = torch.max(disk, torch.zeros_like(disk)).sum()
    return -total_0pers + disk

N = 100
pts = torch.randn(N, 2, requires_grad=True)
rips = Rips(max_edge_length=5, max_dimension=1)
optimizer = torch.optim.Adam([pts], lr=1e-2)

# display original points (color-coded based on position)
with torch.no_grad():
    c = [p[0]**2 + p[1]**2 for p in pts]
    plt.clf()
    plt.scatter(pts[:,0], pts[:,1], c=c, cmap='rainbow')
    plt.show()

# optimize points
losses = []
for _ in range(1000):
    optimizer.zero_grad()
    loss = h_loss(pts, rips)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        losses.append(loss.item())

# plot losses
# plot optimized points with reference circle
with torch.no_grad():
    # plot losses
    plt.clf()
    plt.plot(losses)
    plt.show()

    # optimized points
    plt.clf()
    plt.scatter(pts[:,0], pts[:,1], c=c, cmap='rainbow')

    # reference circle
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x,y)
    F = X**2 + Y**2 - 1
    plt.contour(X,Y,F,[0])

    plt.show()

# figure out distances between nearest points
with torch.no_grad():
    for i in range(len(pts)):
        dists = [torch.norm(pts[i] - pts[j]).item() for j in range(len(pts))]
        dists.sort()
        print(dists[1])
