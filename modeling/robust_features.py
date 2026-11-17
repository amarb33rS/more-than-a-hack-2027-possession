"""A bounded derivative ablation for the existing 66 tree features.

Inputs follow ``common.features``: one complete period, coordinates in metres,
25 Hz, floating-point NaNs for missing observations, and static team membership.
Only ball speed and each team's mean player speed change, including their
existing +/-25-frame context copies. Distances, counts and all other features
remain byte-for-byte unchanged. No labels, status or event fields are accessed.

The default derivative spans five frames (0.2 s). Every observation in that
window must be finite; the derivative never bridges an observation gap. This is
not a tracking-repair algorithm and does not detect identity changes. Reduced
noise on synthetic stationary coordinates does not establish target-match gains.
"""
import numbers
import numpy as np
from .common import features


def _lagged_speed(coordinates, lag, maximum):
    """Endpoint displacement / elapsed seconds, with contiguous finite support."""
    n = len(coordinates)
    speed = np.full(coordinates.shape[:-1], np.nan,
                    dtype=np.result_type(coordinates.dtype, np.float32))
    if n <= lag:
        return speed
    observed = np.isfinite(coordinates).all(-1)
    missing = np.concatenate([
        np.zeros((1, *observed.shape[1:]), dtype=np.int32),
        np.cumsum(~observed, axis=0, dtype=np.int32)], axis=0)
    contiguous = missing[lag+1:] == missing[:-lag-1]
    with np.errstate(invalid='ignore', over='ignore'):
        derivative = (coordinates[lag:]-coordinates[:-lag]) * (25.0/lag)
        magnitude = np.linalg.norm(derivative, axis=-1)
    speed[lag:] = np.where(contiguous & (magnitude <= maximum), magnitude, np.nan)
    return speed


def robust_features(a, lag=5):
    """Return ``(x, names)`` with the exact original 66-column feature contract.

    Call on full periods before row sampling so derivatives and temporal context
    are computed correctly. The first ``lag`` ball speeds are NaN. As in the
    original feature function, a team with no finite player speeds has mean speed
    zero; this ablation adds no availability columns. ``lag=1`` delegates exactly
    to the original implementation, including its dtype and NaN behavior.
    """
    if isinstance(lag, bool) or not isinstance(lag, numbers.Integral) or lag < 1:
        raise ValueError('lag must be a positive integer number of frames')
    lag = int(lag)
    geometry = {key: a[key] for key in ('players', 'ball', 'teams', 'pitch')}
    x, names = features(geometry)
    if lag == 1:
        return x, names
    ball_speed = _lagged_speed(geometry['ball'], lag, 50)
    player_speed = _lagged_speed(geometry['players'], lag, 15)
    replacements = {'ball_speed': ball_speed}
    for team in (0, 1):
        speed = player_speed[:, geometry['teams'] == team]
        replacements[f'{team}_speed'] = (
            np.nansum(speed, axis=1) / np.maximum(np.isfinite(speed).sum(1), 1))
    columns = {name: i for i, name in enumerate(names)}
    idx = np.arange(len(x))
    for name, values in replacements.items():
        x[:, columns[name]] = values
        for offset in (-25, 25):
            x[:, columns[f'{name}_{offset}']] = values[np.clip(idx+offset, 0, len(x)-1)]
    return x, names


def tree_features(a, artifact):
    """Extract the contract recorded in a trusted tree artifact (legacy lag=1).

    Column names alone cannot distinguish derivative lags. Every inference path
    must consult this metadata instead of assuming the original extractor.
    """
    x, names = robust_features(a, lag=artifact.get('velocity_lag', 1))
    if names != artifact['features']:
        raise ValueError('Model feature contract mismatch')
    return x, names
