import matplotlib.pyplot as plt
import torch
from itertools import chain

def get_pts(n_pts, center):
    center = torch.FloatTensor(center)
    return torch.randn(n_pts, 2) + center

def get_xy(pts):
    return pts[:,0], pts[:,1]


n = 100
n_classes = 10
pts = [None] * n_classes
col = [None] * n_classes
for i in range(n_classes):
    pts[i] = get_pts(n, torch.randn(2))
    col[i] = [i / n_classes] * n

pts = torch.cat([p for p in pts], dim=0)
col = list(chain(*[c for c in col]))

plt.scatter(*get_xy(pts), s=10, c=col, cmap='rainbow')
plt.show()
