"""Masked physical motion and offline ball-gap features for one 25-Hz period.

Coordinates must be metres, as in prepared IDSSE and converted target tracking.
Call ``augment_motion`` once after any coordinate corruption, then slice its
precomputed arrays for training/inference chunks. No derivative crosses a period
boundary, and missing positions are never interpolated or treated as observed.
"""
import numpy as np

FPS = 25
PLAYER_MAX_SPEED = 15.0
BALL_MAX_SPEED = 50.0
GAP_CAP_SECONDS = 120.0

PLAYER_MOTION_NAMES = (
    'velocity_x_over_15mps', 'velocity_y_over_15mps', 'velocity_valid',
    'radial_closing_speed_over_65mps', 'closing_valid', 'implausible_jump',
)
BALL_MOTION_NAMES = (
    'velocity_x_over_50mps', 'velocity_y_over_50mps', 'velocity_valid',
    'implausible_jump', 'since_observed_over_120s', 'until_observed_over_120s',
    'previous_observation_exists', 'next_observation_exists',
)
PLAYER_MOTION_DIM = len(PLAYER_MOTION_NAMES)
BALL_MOTION_DIM = len(BALL_MOTION_NAMES)


def _velocity(coordinates, speed_limit, lag=1):
    """Backward differences only across consecutive finite observations."""
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError('Velocity lag must be a positive integer')
    observed = np.isfinite(coordinates).all(-1)
    if lag != 1:
        velocity = np.zeros(coordinates.shape, np.float32)
        consecutive = np.zeros(observed.shape, bool)
        if len(coordinates) > lag:
            missing = np.concatenate([np.zeros((1, *observed.shape[1:]), np.int32),
                                      np.cumsum(~observed, axis=0, dtype=np.int32)])
            support = missing[lag+1:] == missing[:-lag-1]
            with np.errstate(over='ignore', invalid='ignore'):
                np.subtract(coordinates[lag:], coordinates[:-lag], out=velocity[lag:], where=support[..., None])
                velocity *= FPS/lag
            consecutive[lag:] = support
        magnitude = np.hypot(velocity[..., 0], velocity[..., 1])
        jump = consecutive & ((magnitude > speed_limit) | ~np.isfinite(magnitude))
        valid = consecutive & ~jump
        velocity[~valid] = 0
        return velocity, valid, jump
    pair = observed[1:] & observed[:-1]
    velocity = np.zeros(coordinates.shape, np.float32)
    # ``where`` avoids inventing a derivative through NaN/inf coordinates.
    with np.errstate(over='ignore', invalid='ignore'):
        np.subtract(coordinates[1:], coordinates[:-1], out=velocity[1:],
                    where=pair[..., None])
        velocity *= FPS
        speed = np.hypot(velocity[..., 0], velocity[..., 1])
    consecutive = np.zeros(observed.shape, bool)
    consecutive[1:] = pair
    jump = consecutive & ((speed > speed_limit) | ~np.isfinite(speed))
    valid = consecutive & ~jump
    velocity[~valid] = 0
    return velocity, valid, jump


def augment_motion(a, velocity_lag=1):
    """Return a shallow-copy mapping with finite float32 motion feature arrays.

    Added arrays: ``motion_player[T,P,6]`` and ``motion_ball[T,8]``. Existing
    arrays retain their object identities and are not mutated. Only ``players``
    and ``ball`` are read; labels, provider status and event data are ignored.

    Positive radial closing speed means player and ball are approaching. It is
    valid only when both velocities and their current separation are valid.
    Implausible jumps have a separate flag and zero velocity/closing features.
    The first observation after a gap has no valid velocity. Slot-identity jumps
    that happen to be physically plausible cannot be detected by this helper.

    Gap ages use the entire provided period, including future observations for
    offline inference. Ages saturate at 120 s; existence masks distinguish a
    long gap from an observation that has never existed in that direction.
    Recompute after changing coordinates; cached features are not auto-refreshed.
    """
    players, ball = np.asarray(a['players']), np.asarray(a['ball'])
    if players.ndim != 3 or players.shape[-1] != 2 or ball.shape != (len(players), 2):
        raise ValueError('Expected players[T,P,2] and ball[T,2]')
    if len(players) == 0:
        raise ValueError('Expected a nonempty period')
    pv, pm, pj = _velocity(players, PLAYER_MAX_SPEED, velocity_lag)
    bv, bm, bj = _velocity(ball, BALL_MAX_SPEED, velocity_lag)

    pair_seen = np.isfinite(players).all(-1) & np.isfinite(ball).all(-1)[:, None]
    relative = np.zeros(players.shape, np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        np.subtract(players, ball[:, None], out=relative, where=pair_seen[..., None])
        distance = np.hypot(relative[..., 0], relative[..., 1])
    cm = pm & bm[:, None] & pair_seen & np.isfinite(distance) & (distance > 1e-4)
    # Normalize separation before the dot product to keep extreme coordinates
    # from overflowing even when both accepted velocities are bounded.
    unit = np.zeros_like(relative)
    np.divide(relative, distance[..., None], out=unit, where=cm[..., None])
    closing = -(unit*(pv-bv[:, None])).sum(-1)
    closing[~cm] = 0

    pf = np.empty((*players.shape[:2], PLAYER_MOTION_DIM), np.float32)
    pf[..., :2] = np.clip(pv/PLAYER_MAX_SPEED, -1, 1)
    pf[..., 2] = pm
    pf[..., 3] = np.clip(closing/(PLAYER_MAX_SPEED+BALL_MAX_SPEED), -1, 1)
    pf[..., 4] = cm
    pf[..., 5] = pj

    seen = np.isfinite(ball).all(-1)
    idx = np.arange(len(ball), dtype=np.int64)
    previous = np.maximum.accumulate(np.where(seen, idx, -1))
    following = np.minimum.accumulate(np.where(seen, idx, len(ball))[::-1])[::-1]
    has_previous, has_next = previous >= 0, following < len(ball)
    cap_frames = GAP_CAP_SECONDS*FPS
    since = np.where(has_previous, idx-previous, cap_frames)
    until = np.where(has_next, following-idx, cap_frames)
    bf = np.empty((len(ball), BALL_MOTION_DIM), np.float32)
    bf[:, :2] = np.clip(bv/BALL_MAX_SPEED, -1, 1)
    bf[:, 2] = bm
    bf[:, 3] = bj
    bf[:, 4] = np.minimum(since, cap_frames)/cap_frames
    bf[:, 5] = np.minimum(until, cap_frames)/cap_frames
    bf[:, 6] = has_previous
    bf[:, 7] = has_next
    out = dict(a)
    out.update(motion_player=pf, motion_ball=bf, motion_velocity_lag=np.array(velocity_lag, np.int32))
    return out
