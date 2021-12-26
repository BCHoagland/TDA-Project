import torch
from project.models.rips import Rips


class TopAE():
    def __init__(self) -> None:
        self.rips = Rips(max_edge_length=0.5)

    # loss based on 0-dimensional persistent homology death times
    # small death time = bad
    # outside unit disk = bad
    def _top_loss(self, pts):
        deaths = self.rips(pts)
        total_0pers = torch.sum(deaths)
        disk = (pts ** 2).sum(-1) - 1
        disk = torch.max(disk, torch.zeros_like(disk)).sum()
        return -total_0pers + disk

    def loss(self, batch, model, rec_loss, params):
        enc = model.encode(batch)
        out = model.decode(enc)
        rec_loss = rec_loss(out, batch)
        top_loss = self._top_loss(enc)
        return rec_loss + params.top_coef * top_loss
