"""Bounded-memory paired moments and analytic HEALPix NEST latitude bands.

The centered merge and the integer NEST decode follow the verified
paired_input_sky/analyze_full_field.py analysis. The interface here labels
each comparison independently because the reference can be a dirty map.
"""

import math

import numpy as np


REGIONS = ("all", "abs_lat_le_30", "abs_lat_gt_30_le_60", "abs_lat_gt_60")


def compact_even_bits(value):
    value = value & np.uint32(0x55555555)
    value = (value | (value >> 1)) & np.uint32(0x33333333)
    value = (value | (value >> 2)) & np.uint32(0x0F0F0F0F)
    value = (value | (value >> 4)) & np.uint32(0x00FF00FF)
    return (value | (value >> 8)) & np.uint32(0x0000FFFF)


def native_latitude_tags(nside, start, stop):
    ids = np.arange(start, stop, dtype=np.uint32)
    face, local = np.divmod(ids, nside**2)
    ix = compact_even_bits(local).astype(np.int32)
    iy = compact_even_bits(local >> 1).astype(np.int32)
    jrll = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int32)
    jr = jrll[face] * nside - ix - iy - 1
    z = (2 * nside - jr).astype(np.float64) * (2.0 / (3 * nside))
    north = jr < nside
    south = jr > 3 * nside
    z[north] = 1 - jr[north].astype(np.float64)**2 / (3 * nside**2)
    z[south] = -1 + (4 * nside - jr[south]).astype(np.float64)**2 / (3 * nside**2)
    return np.where(np.abs(z) <= 0.5, 0, np.where(np.abs(z) <= math.sqrt(3) / 2, 1, 2))


class Moments:
    def __init__(self):
        self.n = 0
        self.xmean = self.ymean = self.xx = self.yy = self.xy = self.sse = 0.0

    def add(self, reference, candidate):
        n = reference.size
        if not n:
            return
        xmean, ymean = float(reference.mean()), float(candidate.mean())
        dx, dy = reference - xmean, candidate - ymean
        residual = candidate - reference
        count = self.n + n
        k = self.n * n / count
        ax, ay = xmean - self.xmean, ymean - self.ymean
        self.xx += float(dx @ dx) + k * ax**2
        self.yy += float(dy @ dy) + k * ay**2
        self.xy += float(dx @ dy) + k * ax * ay
        self.sse += float(residual @ residual)
        self.xmean += ax * n / count
        self.ymean += ay * n / count
        self.n = count

    def fit(self):
        alpha = self.xy / self.yy if self.yy > 0 else 0.0
        return alpha, self.xmean - alpha * self.ymean

    def result(self, alpha, beta):
        if not self.n:
            return {"pixels": 0}
        sigma = math.sqrt(self.xx / self.n)
        xnorm2 = self.xx + self.n * self.xmean**2
        aligned_mean = alpha * self.ymean + beta - self.xmean
        aligned_sse = max(0.0, self.xx + alpha**2 * self.yy - 2 * alpha * self.xy + self.n * aligned_mean**2)
        raw_rmse = math.sqrt(self.sse / self.n)
        affine_rmse = math.sqrt(aligned_sse / self.n)
        return {
            "pixels": self.n,
            "pearson": self.xy / math.sqrt(self.xx * self.yy) if self.xx > 0 and self.yy > 0 else None,
            "reference_mean": self.xmean,
            "reference_std_population": sigma,
            "candidate_mean": self.ymean,
            "raw_relative_l2": math.sqrt(self.sse / xnorm2) if xnorm2 > 0 else None,
            "raw_rmse": raw_rmse,
            "raw_rmse_over_reference_std": raw_rmse / sigma if sigma > 0 else None,
            "affine_relative_l2": math.sqrt(aligned_sse / xnorm2) if xnorm2 > 0 else None,
            "affine_rmse": affine_rmse,
            "affine_rmse_over_reference_std": affine_rmse / sigma if sigma > 0 else None,
        }


def finalize(moments, pixels):
    if moments[0].n != pixels or sum(moment.n for moment in moments[1:]) != pixels:
        raise ValueError("Native latitude regions do not partition the full domain")
    alpha, beta = moments[0].fit()
    return {
        "global_alpha": alpha,
        "global_beta": beta,
        "regions": {name: moment.result(alpha, beta) for name, moment in zip(REGIONS, moments)},
    }
