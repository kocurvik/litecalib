"""AnyCalib's calibrator (`calibrator_cpu.py`) on the GPU: one cooperative kernel, float32."""

import math
import os
import sys
from pathlib import Path

if not (os.environ.get("CONDA_PREFIX") or os.environ.get("CUDA_HOME")) and (Path(sys.prefix) / "nvvm").is_dir():
    os.environ["CONDA_PREFIX"] = sys.prefix

import numpy as np
import torch
from numba import config as numba_config
from numba import cuda, float32, float64, int32

numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = False

from litecalib import calibrator_cpu as anycalib_nb
from litecalib.calibrator_cpu import DIVISION, KB, PINHOLE, RADIAL, UCM, model_of

THREADS = 256
NWARPS = THREADS // 32
MAXD = 8
NACC = MAXD * MAXD + MAXD + 2
NST = 2 * MAXD + 8
S_P, S_Q, S_COST0, S_COST, S_FX, S_FY, S_AR, S_CX, S_CY, S_OK = (
    0, MAXD, 2 * MAXD, 2 * MAXD + 1, 2 * MAXD + 2, 2 * MAXD + 3, 2 * MAXD + 4, 2 * MAXD + 5,
    2 * MAXD + 6, 2 * MAXD + 7)
PTS_PER_THREAD = 8
MAXD_AUG = MAXD * (MAXD + 1)
MAXD3 = 3 * MAXD

F0, F1, F2, F3, FHALF = (np.float32(v) for v in (0.0, 1.0, 2.0, 3.0, 0.5))
FTHIRD = np.float32(1.0 / 3.0)
FEPS = np.float32(np.finfo(np.float32).eps)
FEPS_PAR = np.float32(anycalib_nb.EPS_PARALLEL)
FTOL = np.float32(anycalib_nb.NEWTON_TOL)
FSQRT3 = np.float32(math.sqrt(3.0))
NEWTON_ITERS = anycalib_nb.NEWTON_ITERS
EPS64 = anycalib_nb.EPS
TAN_HALF_FOV = anycalib_nb.TAN_HALF_FOV


@cuda.jit(device=True)
def _cbrt(x):
    return math.copysign(math.pow(max(abs(x), FEPS), FTHIRD), x)


@cuda.jit(device=True)
def _unproject(model, nk, p, u, v):
    mx = (u - p[2]) / p[0]
    my = (v - p[3]) / p[1]
    if model == PINHOLE:
        n = math.sqrt(mx * mx + my * my + F1)
        return mx / n, my / n, F1 / n, True
    if model == RADIAL:
        rd = math.sqrt(mx * mx + my * my)
        valid = True
        if nk == 1:
            k = p[4]
            k = max(k, FEPS) if k >= F0 else min(k, -FEPS)
            q = F1 / k / F3
            r = FHALF * rd / k
            disc = q * q * q + r * r
            ds = math.sqrt(abs(disc))
            if disc >= F0:
                ru = _cbrt(r + ds) + _cbrt(r - ds)
            else:
                s = _cbrt(math.sqrt(r * r + abs(disc)))
                t = math.atan2(ds, r) / F3
                ru = s * (-math.cos(t) + FSQRT3 * math.sin(t))
        else:
            ru = rd
            f = F0
            for _ in range(NEWTON_ITERS):
                r2 = ru * ru
                ri = r2
                dist = F1 + p[4] * r2
                grad = F1 + F3 * p[4] * r2
                for j in range(1, nk):
                    ri = r2 * ri
                    dist += p[4 + j] * ri
                    grad += float32(3 + 2 * j) * p[4 + j] * ri
                f = ru * dist - rd
                ru = ru - f / grad
            valid = abs(f) <= FTOL
        sc = ru / max(rd, FEPS)
        xx, yy = mx * sc, my * sc
        n = math.sqrt(xx * xx + yy * yy + F1)
        return xx / n, yy / n, F1 / n, valid
    if model == KB:
        r = math.sqrt(mx * mx + my * my)
        th = math.sqrt(r)
        f = F0
        for _ in range(NEWTON_ITERS):
            t2 = th * th
            ti = t2
            fac = F1 + p[4] * t2
            grad = F1 + F3 * p[4] * t2
            for j in range(1, nk):
                ti = ti * t2
                fac += p[4 + j] * ti
                grad += float32(3 + 2 * j) * p[4 + j] * ti
            f = th * fac - r
            th = th - f / grad
        s = math.sin(th) / max(r, FEPS)
        return s * mx, s * my, math.cos(th), abs(f) <= FTOL
    if model == DIVISION:
        r2 = mx * mx + my * my
        ri = r2
        z = F1 + p[4] * r2
        for j in range(1, nk):
            ri = ri * r2
            z += p[4 + j] * ri
        n = math.sqrt(r2 + z * z)
        return mx / n, my / n, z / n, True
    xi = p[4]
    r2 = mx * mx + my * my
    tmp = F1 + (F1 - xi * xi) * r2
    fac = (xi + math.sqrt(max(tmp, F0))) / (F1 + r2)
    return fac * mx, fac * my, fac - xi, tmp >= F0


@cuda.jit(device=True)
def _jac_bearing(model, nk, p, bx, by, bz, u, v, jb, pw):
    """jb: flat (3, MAXD) float32, row a at a * MAXD; pw: float32 scratch (MAXD)."""
    for e in range(3 * MAXD):
        jb[e] = F0
    fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    if model == PINHOLE:
        iz = F1 / max(bz, FEPS)
        px, py = bx * iz, by * iz
        xin = F1 / math.sqrt(px * px + py * py + F1)
        ifx, ify = -F1 / fx, -F1 / fy
        for a in range(3):
            xa = px if a == 0 else (py if a == 1 else F1)
            d0 = xin * ((F1 if a == 0 else F0) - xin * xin * xa * px)
            d1 = xin * ((F1 if a == 1 else F0) - xin * xin * xa * py)
            jb[a * MAXD + 0] = d0 * px * ifx
            jb[a * MAXD + 1] = d1 * py * ify
            jb[a * MAXD + 2] = d0 * ifx
            jb[a * MAXD + 3] = d1 * ify
        return
    if model == RADIAL:
        iz = F1 / max(bz, FEPS)
        pux, puy = bx * iz, by * iz
        ru2 = pux * pux + puy * puy
        ri = ru2
        pw[0] = ru2
        agg1 = F2 * p[4]
        agg2 = p[4] * ru2
        for i in range(1, nk):
            agg1 += float32(2 + 2 * i) * p[4 + i] * ri
            ri = ri * ru2
            pw[i] = ri
            agg2 += p[4 + i] * ri
        a00, a01 = F1 + agg2 + agg1 * pux * pux, agg1 * pux * puy
        a10, a11 = agg1 * puy * pux, F1 + agg2 + agg1 * puy * puy
        m00, m01, m02 = a00 * iz, a01 * iz, -(a00 * pux + a01 * puy) * iz
        m10, m11, m12 = a10 * iz, a11 * iz, -(a10 * pux + a11 * puy) * iz
        g00, g01, g02 = m11 * bz - m12 * by, m12 * bx - m10 * bz, m10 * by - m11 * bx
        g10, g11, g12 = by * m02 - bz * m01, bz * m00 - bx * m02, bx * m01 - by * m00
        den = m00 * g00 + m01 * g01 + m02 * g02
        inx, iny = -F1 / max(fx, FEPS), -F1 / max(fy, FEPS)
        pdx, pdy = (cx - u) * inx, (cy - v) * iny
        for a in range(3):
            ga = (g00 if a == 0 else (g01 if a == 1 else g02)) / den
            gb = (g10 if a == 0 else (g11 if a == 1 else g12)) / den
            jb[a * MAXD + 0] = ga * pdx * inx
            jb[a * MAXD + 1] = gb * pdy * iny
            jb[a * MAXD + 2] = ga * inx
            jb[a * MAXD + 3] = gb * iny
            for i in range(nk):
                jb[a * MAXD + 4 + i] = -(ga * pux * pw[i] + gb * puy * pw[i])
        return
    if model == KB:
        ifx, ify = F1 / fx, F1 / fy
        rr = math.sqrt(bx * bx + by * by)
        th = math.atan2(rr, bz)
        sx, sy = (u - cx) * ifx, (v - cy) * ify
        sir = F1 / max(math.sqrt(sx * sx + sy * sy), FEPS)
        csx, csy = sx * sir, sy * sir
        t2 = th * th
        ti = t2
        dsr = F1 + F3 * p[4] * t2
        pw[0] = t2 * th
        for i in range(1, nk):
            ti = ti * t2
            dsr += float32(3 + 2 * i) * p[4 + i] * ti
            pw[i] = ti * th
        dtdr = F1 / dsr
        drcx, drcy = -ifx * csx, -ify * csy
        for a in range(3):
            dbt = bz * csx if a == 0 else (bz * csy if a == 1 else -rr)
            jb[a * MAXD + 0] = dbt * dtdr * drcx * sx
            jb[a * MAXD + 1] = dbt * dtdr * drcy * sy
            jb[a * MAXD + 2] = dbt * dtdr * drcx
            jb[a * MAXD + 3] = dbt * dtdr * drcy
            for i in range(nk):
                jb[a * MAXD + 4 + i] = dbt * (-pw[i] / dsr)
        for a in range(2):
            for b in range(2):
                ca = csx if a == 0 else csy
                cb = csx if b == 0 else csy
                ifb = ifx if b == 0 else ify
                sb = sx if b == 0 else sy
                dcdc = ifb * sir * (ca * cb - (F1 if a == b else F0))
                jb[a * MAXD + b] += rr * sb * dcdc
                jb[a * MAXD + 2 + b] += rr * dcdc
        return
    if model == DIVISION:
        nix, niy = -F1 / fx, -F1 / fy
        mx, my = (cx - u) * nix, (cy - v) * niy
        mnx, mny = mx * nix, my * niy
        r2 = mx * mx + my * my
        ri = r2
        mz = F1 + p[4] * r2
        dmz = p[4]
        for i in range(1, nk):
            dmz += float32(i + 1) * p[4 + i] * ri
            ri = ri * r2
            mz += p[4 + i] * ri
        inorm = F1 / (r2 + mz * mz)
        sq = math.sqrt(inorm)
        for a in range(3):
            ma = mx if a == 0 else (my if a == 1 else mz)
            d0 = sq * ((F1 if a == 0 else F0) - inorm * ma * mx)
            d1 = sq * ((F1 if a == 1 else F0) - inorm * ma * my)
            d2 = sq * ((F1 if a == 2 else F0) - inorm * ma * mz)
            jb[a * MAXD + 0] = d0 * mnx + d2 * dmz * F2 * mx * mnx
            jb[a * MAXD + 1] = d1 * mny + d2 * dmz * F2 * my * mny
            jb[a * MAXD + 2] = d0 * nix + d2 * dmz * F2 * mnx
            jb[a * MAXD + 3] = d1 * niy + d2 * dmz * F2 * mny
            pk = r2
            for i in range(nk):
                jb[a * MAXD + 4 + i] = d2 * pk
                pk = pk * r2
        return
    ifx, ify = F1 / fx, F1 / fy
    xi = p[4]
    mx, my = (u - cx) * ifx, (v - cy) * ify
    r2 = mx * mx + my * my
    ir2p1 = F1 / (r2 + F1)
    tmp = math.sqrt(max(F1 + (F1 - xi * xi) * r2, F0))
    fac = (xi + tmp) * ir2p1
    dfac = ir2p1 * (FHALF * (F1 - xi * xi) / tmp - fac)
    dcx, dcy = -F2 * mx * ifx, -F2 * my * ify
    for a in range(3):
        ma = mx if a == 0 else (my if a == 1 else F1)
        jb[a * MAXD + 0] = dfac * ma * mx * dcx
        jb[a * MAXD + 1] = dfac * ma * my * dcy
        jb[a * MAXD + 2] = dfac * ma * dcx
        jb[a * MAXD + 3] = dfac * ma * dcy
    jb[0] += FHALF * fac * dcx
    jb[MAXD + 1] += FHALF * fac * dcy
    jb[2] -= fac * ifx
    jb[MAXD + 3] -= fac * ify
    dxi = (tmp - r2 * xi) * ir2p1 / tmp
    s = F2 * math.sqrt(xi)
    jb[4] = mx * dxi * s
    jb[MAXD + 4] = my * dxi * s
    jb[2 * MAXD + 4] = (dxi - F1) * s


@cuda.jit(device=True)
def _logmap(x0, x1, x2, y0, y1, y2, g):
    """g: flat (2, 3) float32. Returns (r0, r1, valid)."""
    pole = abs(x1) >= F1 - FEPS
    e00, e02 = (F1, F0) if pole else (x2, -x0)
    norm = math.sqrt(e00 * e00 + e02 * e02)
    e00, e02 = e00 / norm, e02 / norm
    e10, e11, e12 = x1 * e02, x2 * e00 - x0 * e02, -x1 * e00
    cos_t = x0 * y0 + x1 * y1 + x2 * y2
    npar = cos_t < F1 - FEPS_PAR
    d0, d1, d2 = x0 - y0, x1 - y1, x2 - y2
    chord = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    t = F2 * math.asin(min(max(FHALF * chord, F0), F1 - FEPS))
    by0 = e00 * y0 + e02 * y2
    by1 = e10 * y0 + e11 * y1 + e12 * y2
    sc = t / math.sin(t) if npar else F1
    inv_cub = F1 / max(e11 * e11 * e11, FEPS)
    c = cos_t if npar else F0
    cosec2 = F1 / (F1 - c * c)
    tcs = t * math.sqrt(cosec2)
    t1 = cosec2 * (F1 - c * tcs) if npar else FTHIRD
    xy = x0 * y0 + x2 * y2
    x012 = x0 * x1 * x2
    s2 = tcs if npar else F1
    g[0] = t1 * by0 * d0 + s2 * (-x2 * xy * inv_cub)
    g[1] = t1 * by0 * d1
    g[2] = t1 * by0 * d2 + s2 * (x0 * xy * inv_cub)
    g[3] = t1 * by1 * d0 + s2 * ((x0 * x0 * x0 * y1 + x012 * y2 + x0 * x2 * x2 * y1 - x1 * x2 * x2 * y0) * inv_cub)
    g[4] = t1 * by1 * d1 + s2 * (-xy / max(e11, FEPS))
    g[5] = t1 * by1 * d2 + s2 * ((-(x0 * x0) * x1 * y2 + x0 * x0 * x2 * y1 + x012 * y0 + x2 * x2 * x2 * y1) * inv_cub)
    return sc * by0, sc * by1, inv_cub < F1 / FEPS


@cuda.jit(device=True)
def _block_sum_to(acc, n, warp_buf, part, row):
    """part[row, e] = the block's sum of acc[e] (float64), e < n; thread 0 writes."""
    lane = cuda.threadIdx.x & 31
    wid = cuda.threadIdx.x >> 5
    for e in range(n):
        v = float64(acc[e])
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 16)
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 8)
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 4)
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 2)
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 1)
        if lane == 0:
            warp_buf[wid, e] = v
    cuda.syncthreads()
    t = cuda.threadIdx.x
    if t < n:
        s = 0.0
        for j in range(NWARPS):
            s += warp_buf[j, t]
        part[row, t] = s
    cuda.syncthreads()


@cuda.jit(device=True)
def _block_max2_to(a, b, warp_buf, part, row):
    lane = cuda.threadIdx.x & 31
    wid = cuda.threadIdx.x >> 5
    for e in range(2):
        v = a if e == 0 else b
        for off in (16, 8, 4, 2, 1):
            v = max(v, cuda.shfl_down_sync(0xFFFFFFFF, v, off))
        if lane == 0:
            warp_buf[wid, e] = v
    cuda.syncthreads()
    t = cuda.threadIdx.x
    if t < 2:
        m = warp_buf[0, t]
        for j in range(1, NWARPS):
            m = max(m, warp_buf[j, t])
        part[row, t] = m
    cuda.syncthreads()


@cuda.jit(device=True)
def _gather(part, b, chunks, n, out):
    """out[e] = sum over the image's chunks of part[., e]."""
    for e in range(n):
        out[e] = 0.0
    for c in range(chunks):
        for e in range(n):
            out[e] += part[b * chunks + c, e]


@cuda.jit(device=True)
def _solve(a, rhs, out, n, scaled):
    """Gaussian elimination with partial pivoting on flat (n, n) float64 ``a``; with `scaled`, Jacobi column
    scaling first. False if singular.
    """
    m = cuda.local.array(MAXD_AUG, float64)
    sc = cuda.local.array(MAXD, float64)
    w = n + 1
    for i in range(n):
        sc[i] = 1.0 / math.sqrt(a[i * n + i]) if (scaled and a[i * n + i] > 0) else 1.0
    for i in range(n):
        for j in range(n):
            m[i * w + j] = a[i * n + j] * sc[i] * sc[j]
        m[i * w + n] = rhs[i] * sc[i]
    for col in range(n):
        piv = col
        for i in range(col + 1, n):
            if abs(m[i * w + col]) > abs(m[piv * w + col]):
                piv = i
        pv = m[piv * w + col]
        if pv == 0.0 or not math.isfinite(pv):
            return False
        if piv != col:
            for j in range(w):
                tmp = m[col * w + j]
                m[col * w + j] = m[piv * w + j]
                m[piv * w + j] = tmp
        for i in range(col + 1, n):
            fac = m[i * w + col] / m[col * w + col]
            for j in range(col, w):
                m[i * w + j] -= fac * m[col * w + j]
    for i in range(n - 1, -1, -1):
        acc = m[i * w + n]
        for j in range(i + 1, n):
            acc -= m[i * w + j] * out[j]
        out[i] = acc / m[i * w + i]
    for i in range(n):
        out[i] *= sc[i]
    return True


@cuda.jit(device=True)
def _solve_tied(tot, d, hold, sol):
    """anycalib_nb._solve_tied: the Gauss-Newton system of flat ``tot`` (J'J then -J'r) in the free parameters
    only -- fx and fy one unknown f (upstream's simple_* parameterization: the step of f is the step of both,
    its Jacobian the sum of theirs), cx, cy dropped when `hold` -- the step expanded back into ``sol`` (d).
    False if singular.
    """
    grp = cuda.local.array(MAXD, int32)
    m = 0
    for a in range(d):
        if a == 1:
            grp[a] = grp[0]
        elif (a == 2 or a == 3) and hold:
            grp[a] = -1
        else:
            grp[a] = m
            m += 1
    ar = cuda.local.array(NACC, float64)
    br = cuda.local.array(MAXD, float64)
    sr = cuda.local.array(MAXD, float64)
    for e in range(m * m):
        ar[e] = 0.0
    for e in range(m):
        br[e] = 0.0
    for a in range(d):
        ga = grp[a]
        if ga < 0:
            continue
        br[ga] += tot[d * d + a]
        for bb in range(d):
            gb = grp[bb]
            if gb >= 0:
                ar[ga * m + gb] += tot[a * d + bb]
    ok = _solve(ar, br, sr, m, False)
    for a in range(d):
        sol[a] = 0.0 if grp[a] < 0 else sr[grp[a]]
    return ok


@cuda.jit(cache=True)
def _calib_kernel(model, nk, x, rays, iters, cxcy, hold, simple, chunks, part, st, out):
    """anycalib_nb._calibrate_one for image ``blockIdx.x // chunks``, its points spread over `chunks` blocks;
    grid-wide barriers between the blocks' sums and the leader's solves. `hold`: ``c = cxcy[b]`` given (Eq.
    11's replaced) and kept by Gauss-Newton. `simple` (with `hold`): upstream's ``simple_<model>``, one focal
    for both axes (anycalib_nb's `simple`).
    """
    grid = cuda.cg.this_grid()
    blk = cuda.blockIdx.x
    b = blk // chunks
    c = blk % chunks
    tid = cuda.threadIdx.x
    leader = c == 0 and tid == 0
    n = x.shape[0]
    d = 4 + nk
    nacc = d * d + d + 2
    stride = chunks * THREADS
    start = c * THREADS + tid
    warp_buf = cuda.shared.array((NWARPS, NACC), float64)
    acc = cuda.local.array(NACC, float64)
    tot = cuda.local.array(NACC, float64)
    sol = cuda.local.array(MAXD, float64)
    row = cuda.local.array(MAXD, float64)

    if model == UCM or model == DIVISION:
        for e in range(12):
            acc[e] = 0.0
        for i in range(start, n, stride):
            X, Y = float64(rays[b, i, 0]), float64(rays[b, i, 1])
            r0, r1, r2_ = Y * float64(x[i, 0]), -Y, X
            t = float64(x[i, 1]) * X
            acc[0] += r0 * r0
            acc[1] += r0 * r1
            acc[2] += r0 * r2_
            acc[3] += r1 * r0
            acc[4] += r1 * r1
            acc[5] += r1 * r2_
            acc[6] += r2_ * r0
            acc[7] += r2_ * r1
            acc[8] += r2_ * r2_
            acc[9] += r0 * t
            acc[10] += r1 * t
            acc[11] += r2_ * t
        _block_sum_to(acc, 12, warp_buf, part, blk)
        grid.sync()
        if leader:
            _gather(part, b, chunks, 12, tot)
            ok = _solve(tot, tot[9:], sol, 3, False)
            st[b, S_AR], st[b, S_CX], st[b, S_CY] = sol[0], sol[1] / sol[0], sol[2]
            if hold:
                st[b, S_CX], st[b, S_CY] = cxcy[b, 0], cxcy[b, 1]
            st[b, S_OK] = 1.0 if ok else 0.0
            if simple:
                st[b, S_AR], st[b, S_OK] = 1.0, 1.0
        grid.sync()
        ar, cx, cy = st[b, S_AR], st[b, S_CX], st[b, S_CY]
        if model == DIVISION:
            q = 1 + nk
            for e in range(q * q + q):
                acc[e] = 0.0
            for i in range(start, n, stride):
                X, Y, Z = float64(rays[b, i, 0]), float64(rays[b, i, 1]), float64(rays[b, i, 2])
                xc, yc = float64(x[i, 0]) - cx, float64(x[i, 1]) - cy
                rc = math.hypot(xc, yc)
                ra = math.hypot(X, ar * Y)
                rca2 = xc * xc + yc * yc / (ar * ar)
                row[0] = ra
                row[1] = ra * rca2
                for j in range(2, q):
                    row[j] = row[j - 1] * rca2
                t = rc * Z
                for a in range(q):
                    acc[q * q + a] += row[a] * t
                    for bb in range(q):
                        acc[a * q + bb] += row[a] * row[bb]
            _block_sum_to(acc, q * q + q, warp_buf, part, blk)
            grid.sync()
            if leader:
                _gather(part, b, chunks, q * q + q, tot)
                ok = _solve(tot, tot[q * q:], sol, q, False) and st[b, S_OK] == 1.0
                fx = sol[0]
                st[b, S_P + 0], st[b, S_P + 1], st[b, S_P + 2], st[b, S_P + 3] = fx, ar * fx, cx, cy
                fp = fx
                for j in range(nk):
                    st[b, S_P + 4 + j] = sol[1 + j] * fp
                    fp = fp * fx * fx
                st[b, S_OK] = 1.0 if ok else 0.0
            grid.sync()
        else:
            for e in range(8):
                acc[e] = 0.0
            for i in range(start, n, stride):
                X, Y, Z = float64(rays[b, i, 0]), float64(rays[b, i, 1]), float64(rays[b, i, 2])
                rc = math.hypot(float64(x[i, 0]) - cx, float64(x[i, 1]) - cy)
                ra = math.hypot(X, ar * Y)
                dd = math.sqrt(X * X + Y * Y + Z * Z)
                q0, q1 = ra, -rc * dd
                t = rc * Z
                acc[0] += q0 * q0
                acc[1] += q0 * q1
                acc[2] += q1 * q0
                acc[3] += q1 * q1
                acc[4] += q0 * t
                acc[5] += q1 * t
                if ra > EPS64 and Z > EPS64:
                    acc[6] += rc * Z / max(ra, EPS64)
                    acc[7] += 1.0
            _block_sum_to(acc, 8, warp_buf, part, blk)
            grid.sync()
            if leader:
                _gather(part, b, chunks, 8, tot)
                ok = _solve(tot, tot[4:], sol, 2, False) and st[b, S_OK] == 1.0
                fx, xi = sol[0], sol[1]
                if xi < 0:
                    fx, xi = tot[6] / max(tot[7], 1.0), 0.0
                st[b, S_P + 0], st[b, S_P + 1], st[b, S_P + 2], st[b, S_P + 3], st[b, S_P + 4] = (
                    fx, ar * fx, cx, cy, xi)
                st[b, S_OK] = 1.0 if ok else 0.0
            grid.sync()
    else:
        fac_x, fac_y = 1.0, 1.0
        if model != KB:
            mx, my = -math.inf, -math.inf
            for i in range(start, n, stride):
                mx = max(mx, float64(x[i, 0]))
                my = max(my, float64(x[i, 1]))
            _block_max2_to(mx, my, warp_buf, part, blk)
            grid.sync()
            if leader:
                mx, my = -math.inf, -math.inf
                for cc in range(chunks):
                    mx = max(mx, part[b * chunks + cc, 0])
                    my = max(my, part[b * chunks + cc, 1])
                st[b, S_FX], st[b, S_FY] = mx, my
            grid.sync()
            fac_x, fac_y = st[b, S_FX], st[b, S_FY]
            if simple:
                fac_x = fac_y = max(fac_x, fac_y)
        for e in range(d * d + d):
            acc[e] = 0.0
        for i in range(start, n, stride):
            X, Y, Z = float64(rays[b, i, 0]), float64(rays[b, i, 1]), float64(rays[b, i, 2])
            if model == KB:
                rr = math.hypot(X, Y)
                th = math.atan2(rr, Z)
                ox, oy = th * X / max(rr, EPS64), th * Y / max(rr, EPS64)
                w2 = th * th
            else:
                if not math.hypot(X, Y) < Z * TAN_HALF_FOV:
                    continue
                iz = 1.0 / max(Z, EPS64)
                ox, oy = X * iz, Y * iz
                w2 = ox * ox + oy * oy
            for axis in range(2):
                o = ox if axis == 0 else oy
                for j in range(d):
                    row[j] = 0.0
                if hold:
                    row[0 if simple else axis] = (float64(x[i, axis]) - cxcy[b, axis]) / (fac_x if axis == 0 else fac_y)
                else:
                    row[axis] = float64(x[i, axis]) / (fac_x if axis == 0 else fac_y)
                    row[2 + axis] = -1.0
                if model != PINHOLE:
                    cc_ = -o * w2
                    for j in range(nk):
                        row[4 + j] = cc_
                        cc_ *= w2
                for a in range(d):
                    acc[d * d + a] += row[a] * o
                    for bb in range(d):
                        acc[a * d + bb] += row[a] * row[bb]
        _block_sum_to(acc, d * d + d, warp_buf, part, blk)
        grid.sync()
        if leader:
            _gather(part, b, chunks, d * d + d, tot)
            if hold:
                for j in range(d):
                    tot[2 * d + j], tot[3 * d + j] = 0.0, 0.0
                    tot[j * d + 2], tot[j * d + 3] = 0.0, 0.0
                tot[2 * d + 2], tot[3 * d + 3] = 1.0, 1.0
                tot[d * d + 2], tot[d * d + 3] = 0.0, 0.0
                if simple:
                    for j in range(d):
                        tot[1 * d + j], tot[j * d + 1] = 0.0, 0.0
                    tot[1 * d + 1], tot[d * d + 1] = 1.0, 0.0
            ok = _solve(tot, tot[d * d:], sol, d, model == KB)
            if model != KB:
                for j in range(d):
                    if not math.isfinite(sol[j]):
                        sol[j] = 1.0
            fx = fac_x / sol[0]
            fy = fx if simple else fac_y / sol[1]
            st[b, S_P + 0], st[b, S_P + 1] = fx, fy
            if hold:
                st[b, S_P + 2], st[b, S_P + 3] = cxcy[b, 0], cxcy[b, 1]
            else:
                st[b, S_P + 2], st[b, S_P + 3] = sol[2] * fx, sol[3] * fy
            for j in range(nk):
                st[b, S_P + 4 + j] = sol[4 + j]
            st[b, S_OK] = 1.0 if ok else 0.0
        grid.sync()

    if leader:
        for j in range(d):
            st[b, S_Q + j] = st[b, S_P + j]
    grid.sync()
    p32 = cuda.local.array(MAXD, float32)
    acc32 = cuda.local.array(NACC, float32)
    jb = cuda.local.array(MAXD3, float32)
    pw = cuda.local.array(MAXD, float32)
    g = cuda.local.array(6, float32)
    jr0 = cuda.local.array(MAXD, float32)
    jr1 = cuda.local.array(MAXD, float32)
    for it in range(iters + 1):
        for j in range(d):
            p32[j] = float32(st[b, S_Q + j])
        for e in range(nacc):
            acc32[e] = F0
        for i in range(start, n, stride):
            u, v = x[i, 0], x[i, 1]
            bx, by, bz, ok1 = _unproject(model, nk, p32, u, v)
            r0, r1, ok2 = _logmap(bx, by, bz, rays[b, i, 0], rays[b, i, 1], rays[b, i, 2], g)
            if not (ok1 and ok2):
                continue
            _jac_bearing(model, nk, p32, bx, by, bz, u, v, jb, pw)
            for cc in range(d):
                jr0[cc] = g[0] * jb[cc] + g[1] * jb[MAXD + cc] + g[2] * jb[2 * MAXD + cc]
                jr1[cc] = g[3] * jb[cc] + g[4] * jb[MAXD + cc] + g[5] * jb[2 * MAXD + cc]
            for a in range(d):
                acc32[d * d + a] -= jr0[a] * r0 + jr1[a] * r1
                for bb in range(d):
                    acc32[a * d + bb] += jr0[a] * jr0[bb] + jr1[a] * jr1[bb]
            acc32[d * d + d] += r0 * r0 + r1 * r1
            acc32[d * d + d + 1] += F1
        _block_sum_to(acc32, nacc, warp_buf, part, blk)
        grid.sync()
        if leader:
            _gather(part, b, chunks, nacc, tot)
            cost = tot[d * d + d] / tot[d * d + d + 1]
            if it == 0:
                st[b, S_COST0] = cost
            st[b, S_COST] = cost
            if it < iters:
                if simple:
                    solved = _solve_tied(tot, d, hold, sol)
                else:
                    if hold:
                        for j in range(d):
                            tot[2 * d + j], tot[3 * d + j] = 0.0, 0.0
                            tot[j * d + 2], tot[j * d + 3] = 0.0, 0.0
                        tot[2 * d + 2], tot[3 * d + 3] = 1.0, 1.0
                        tot[d * d + 2], tot[d * d + 3] = 0.0, 0.0
                    solved = _solve(tot, tot[d * d:], sol, d, False)
                if not solved:
                    for j in range(d):
                        sol[j] = 0.0
                for j in range(d):
                    if not math.isfinite(sol[j]):
                        sol[j] = 0.0
                for j in range(4):
                    st[b, S_Q + j] += sol[j]
                if model == UCM:
                    xi = st[b, S_Q + 4]
                    st[b, S_Q + 4] = xi + 2.0 * math.sqrt(xi) * sol[4] + sol[4] * sol[4]
                else:
                    for j in range(4, d):
                        st[b, S_Q + j] += sol[j]
        grid.sync()
    if leader:
        src = S_Q if st[b, S_COST] < st[b, S_COST0] else S_P
        for j in range(d):
            out[b, j] = st[b, src + j]


_MAX_BLOCKS: dict = {}
_X_CACHE: dict = {}


def _max_blocks(device) -> int:
    """How many blocks of `_calib_kernel` can be co-resident (a cooperative launch's limit)."""
    if device not in _MAX_BLOCKS:
        overload = next(iter(_calib_kernel.overloads.values()))
        _MAX_BLOCKS[device] = overload.max_cooperative_grid_blocks(THREADS)
    return _MAX_BLOCKS[device]


def fit_rays(cam_id: str, rays: torch.Tensor, x, iters: int = 10, cxcy=None,
             simple: bool = False) -> torch.Tensor:
    """AnyCalib's calibrator on the GPU: (B, N, 3) CUDA rays (float32) at the pixels ``x`` (N, 2, numpy or
    tensor) -> (B, 4 + nk) float64 intrinsics on the rays' device. ``cxcy`` (B, 2) or (2,): the principal
    point given in the field's frame and held (not kb). ``simple`` (with ``cxcy``): upstream's
    ``simple_<model>``, one focal for both axes, reported as ``fx == fy``.
    """
    model, nk = model_of(cam_id)
    bsz, n = rays.shape[0], rays.shape[1]
    dev = rays.device
    if isinstance(cxcy, torch.Tensor) and cxcy.device == dev:
        anycalib_nb.held_cxcy(model, cam_id, np.zeros(2), bsz)
        cd, hold = cxcy.to(torch.float64).reshape(-1, 2).expand(bsz, 2).contiguous(), True
    else:
        c, hold = anycalib_nb.held_cxcy(model, cam_id, cxcy, bsz)
        cd = torch.from_numpy(c).to(dev)
    if simple and not hold:
        raise ValueError("simple (one focal) goes with a held c, as upstream's simple_* models")
    key = (dev, id(x) if not isinstance(x, torch.Tensor) else x.data_ptr(), n)
    if key not in _X_CACHE:
        xt = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x, dtype=torch.float32)
        _X_CACHE[key] = (x, xt.to(dev).contiguous())
    xd = _X_CACHE[key][1]
    rays32 = rays.detach().to(torch.float32).contiguous()
    want = max(1, -(-n // (THREADS * PTS_PER_THREAD)))
    part = torch.empty((1, NACC), dtype=torch.float64, device=dev)
    st = torch.empty((bsz, NST), dtype=torch.float64, device=dev)
    out = torch.empty((bsz, 4 + nk), dtype=torch.float64, device=dev)
    if not bsz:
        return out
    with torch.cuda.device(dev):
        stream = cuda.external_stream(torch.cuda.current_stream().cuda_stream)
        args = (model, nk, cuda.as_cuda_array(xd), cuda.as_cuda_array(rays32), int(iters),
                cuda.as_cuda_array(cd), bool(hold), bool(simple))
        if not _calib_kernel.overloads:
            part1 = torch.empty((bsz, NACC), dtype=torch.float64, device=dev)
            _calib_kernel[bsz, THREADS, stream](*args, 1, cuda.as_cuda_array(part1),
                                                cuda.as_cuda_array(st), cuda.as_cuda_array(out))
        chunks = max(1, min(want, _max_blocks(dev) // bsz))
        if bsz * chunks > _max_blocks(dev):
            raise ValueError(f"a batch of {bsz} images exceeds a cooperative launch ({_max_blocks(dev)} blocks)")
        part = torch.empty((bsz * chunks, NACC), dtype=torch.float64, device=dev)
        _calib_kernel[bsz * chunks, THREADS, stream](*args, chunks, cuda.as_cuda_array(part),
                                                     cuda.as_cuda_array(st), cuda.as_cuda_array(out))
    return out
