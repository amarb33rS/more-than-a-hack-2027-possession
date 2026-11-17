"""Permutation-invariant team encoder and offline temporal convolutions."""
import numpy as np
import torch
from torch import nn

HALO = 63  # exact receptive-field radius: sum(1,2,4,8,16,32)


def swap_team_labels(values):
    """Exchange H/A (or team IDs 0/1), preserving dead/unknown/missing IDs."""
    return torch.where((values >= 0) & (values < 2), 1-values, values)


def tensors(a, start=0, stop=None):
    stop = len(a['ball']) if stop is None else stop
    p = a['players'][start:stop] / a['pitch']
    b = a['ball'][start:stop] / a['pitch']
    pm = np.isfinite(p).all(-1)
    bm = np.isfinite(b).all(-1)
    rel = p - b[:, None]
    # Explicit observation masks prevent missing coordinates becoming fake objects.
    pf = np.concatenate([np.nan_to_num(p), np.nan_to_num(rel),
                         pm[..., None], np.broadcast_to(bm[:, None, None], (*pm.shape, 1))], -1)
    bf = np.concatenate([np.nan_to_num(b), bm[:, None]], -1)
    if 'motion_player' in a:
        pf = np.concatenate([pf, a['motion_player'][start:stop]], -1)
        bf = np.concatenate([bf, a['motion_ball'][start:stop]], -1)
    return (torch.tensor(pf, dtype=torch.float32), torch.tensor(bf, dtype=torch.float32),
            torch.tensor(a['teams'], dtype=torch.long), torch.tensor(pm))


class PossessionNet(nn.Module):
    def __init__(self, width=32, architecture='set-tcn-v1'):
        super().__init__()
        if architecture not in ('set-tcn-v1', 'set-tcn-geometry-v2', 'set-tcn-motion-v3', 'set-tcn-motion-robust-v4'):
            raise ValueError(f'Unknown architecture: {architecture}')
        self.architecture = architecture
        player_extra, ball_extra = 0, 0
        if architecture in ('set-tcn-motion-v3', 'set-tcn-motion-robust-v4'):
            from .motion import PLAYER_MOTION_DIM, BALL_MOTION_DIM
            player_extra, ball_extra = PLAYER_MOTION_DIM, BALL_MOTION_DIM
        self.player = nn.Sequential(nn.Linear(6+player_extra, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.input = nn.Conv1d(width * 4 + 5 + (8 if architecture != 'set-tcn-v1' else 0) + ball_extra, width, 1)
        self.layers = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d)
                                     for d in (1, 2, 4, 8, 16, 32)])
        self.state = nn.Conv1d(width, 3, 1)
        self.boundary = nn.Conv1d(width, 1, 1)

    def forward(self, players, ball, teams, mask):
        # B,T,P,F; teams B,P. No slot-dependent parameters.
        e = self.player(players)
        pooled = []
        counts = []
        geometry = []
        for team in (0, 1):
            valid = mask & (teams[:, None, :] == team)
            count = valid.sum(-1, keepdim=True)
            mean = (e * valid[..., None]).sum(-2) / count.clamp_min(1)
            maximum = e.masked_fill(~valid[..., None], -1e4).max(-2).values
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
            pooled.extend([mean, maximum]); counts.append(count / 11)
            if self.architecture != 'set-tcn-v1':
                # Distances in pitch-normalized coordinates; no label/status inputs.
                observed = valid & (ball[..., 2:3] > 0)
                distance = players[..., 2:4].norm(dim=-1)
                distance = distance.masked_fill(~observed, 2.0)
                # Pad for periods with fewer than three player slots.
                distance = torch.cat([distance, torch.full_like(distance[..., :1], 2.0).expand(*distance.shape[:-1], 3)], -1)
                nearest = distance.topk(3, largest=False).values
                close = ((distance < .03).sum(-1, keepdim=True) / 11)
                geometry.extend([nearest, close])
        h = torch.cat(pooled + [ball] + counts + geometry, -1).transpose(1, 2)
        h = torch.relu(self.input(h))
        for layer in self.layers:
            h = h + torch.relu(layer(h))
        return self.state(h).transpose(1, 2), self.boundary(h).squeeze(1)


@torch.no_grad()
def predict(model, a, device='cpu', chunk=1024):
    model.eval()
    if model.architecture in ('set-tcn-motion-v3', 'set-tcn-motion-robust-v4'):
        lag = 5 if model.architecture == 'set-tcn-motion-robust-v4' else 1
        if 'motion_player' not in a or int(a.get('motion_velocity_lag', 1)) != lag:
            from .motion import augment_motion
            a = augment_motion(a, velocity_lag=lag)
    probabilities, boundaries = [], []
    n = len(a['ball'])
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        left, right = max(0, lo-HALO), min(n, hi+HALO)
        inputs = [t.unsqueeze(0).to(device) for t in tensors(a, left, right)]
        logits, boundary = model(*inputs)
        cut = slice(lo-left, hi-left)
        probabilities.append(logits[0, cut].softmax(-1).cpu().numpy())
        boundaries.append(boundary[0, cut].sigmoid().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(boundaries)
