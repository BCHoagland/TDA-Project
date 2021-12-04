# Test to show that sampling from a Gaussian overshoots the unit ball

import matplotlib.pyplot as plt
import torch

def normal(n, d):
    return torch.randn(n, d)

def ball(n, d):
    N = torch.randn(n, d)
    norm = N.norm(dim=-1).unsqueeze(dim=-1).repeat(1, d)
    N = N / norm
    U = torch.rand(n, 1).repeat(1,d)
    return N * (U**(1/d))

def get_xy(pts):
    return pts[:,0], pts[:,1]


n = 100
ball_pts = ball(n, 2)
normal_pts = normal(n, 2)
x, y = get_xy(torch.cat((ball_pts, normal_pts), dim=0))

ball_c = [0.0] * len(ball_pts)
normal_c = [1.0] * len(normal_pts)
c = ball_c + normal_c

plt.scatter(x, y, s=10, c=c, cmap='rainbow')
plt.show()
# scatter(ball(100, d=2), show=False, c=0.0, label='ball')
# scatter(normal(100, d=2), c=1.0, label='normal')
