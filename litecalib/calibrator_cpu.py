"""AnyCalib's calibrator ported to Numba (CPU, float64): the same algorithm, compiled."""

import math

import numpy as np
from numba import njit, prange

from litecalib.geometry import parse_cam_id

PINHOLE, RADIAL, KB, UCM, DIVISION = range(5)
MODELS = {"pinhole": PINHOLE, "radial": RADIAL, "kb": KB, "ucm": UCM, "division": DIVISION}
EPS = float(np.finfo(np.float32).eps)
EPS_PARALLEL = 1e-4
NEWTON_ITERS = 25
NEWTON_TOL = 1e-5
TAN_HALF_FOV = math.tan(0.5 * math.radians(170.0))
PARALLEL_MIN = 4096
N_CHUNKS = 64


def model_of(cam_id: str) -> tuple[int, int]:
    name, nk = parse_cam_id(cam_id)
    if name not in MODELS:
        raise ValueError(f"{cam_id!r}: AnyCalib's calibrator is ported for {sorted(MODELS)} only")
    if name == "ucm":
        nk = 1
    return MODELS[name], nk


@njit(cache=True, error_model="numpy")
def _solve(a, b, out):
    """Gaussian elimination with partial pivoting (torch.linalg.solve_ex's LU); False if singular. ``a`` (n, n)
    and ``b`` (n,) are not modified.
    """
    n = len(b)
    m = np.empty((n, n + 1))
    for i in range(n):
        for j in range(n):
            m[i, j] = a[i, j]
        m[i, n] = b[i]
    for col in range(n):
        piv = col
        for i in range(col + 1, n):
            if abs(m[i, col]) > abs(m[piv, col]):
                piv = i
        if m[piv, col] == 0.0 or not np.isfinite(m[piv, col]):
            return False
        if piv != col:
            for j in range(n + 1):
                m[col, j], m[piv, j] = m[piv, j], m[col, j]
        for i in range(col + 1, n):
            fac = m[i, col] / m[col, col]
            for j in range(col, n + 1):
                m[i, j] -= fac * m[col, j]
    for i in range(n - 1, -1, -1):
        acc = m[i, n]
        for j in range(i + 1, n):
            acc -= m[i, j] * out[j]
        out[i] = acc / m[i, i]
    return True


@njit(cache=True, error_model="numpy")
def _solve_scaled(ata, atb, out):
    """The normal equations with Jacobi column scaling (for kb's ill-conditioned system)."""
    n = len(atb)
    s = np.empty(n)
    for i in range(n):
        s[i] = 1.0 / math.sqrt(ata[i, i]) if ata[i, i] > 0 else 1.0
    a = np.empty((n, n))
    b = np.empty(n)
    for i in range(n):
        b[i] = atb[i] * s[i]
        for j in range(n):
            a[i, j] = ata[i, j] * s[i] * s[j]
    ok = _solve(a, b, out)
    for i in range(n):
        out[i] *= s[i]
    return ok


@njit(cache=True, error_model="numpy")
def _cbrt(x):
    return math.copysign(max(abs(x), EPS) ** (1.0 / 3.0), x)


@njit(cache=True, error_model="numpy")
def _unproject(model, nk, p, u, v):
    """cam.unproject for one pixel: (bx, by, bz, valid)."""
    mx = (u - p[2]) / p[0]
    my = (v - p[3]) / p[1]
    if model == PINHOLE:
        n = math.sqrt(mx * mx + my * my + 1.0)
        return mx / n, my / n, 1.0 / n, True
    if model == RADIAL:
        rd = math.hypot(mx, my)
        valid = True
        if nk == 1:
            k = p[4]
            k = max(k, EPS) if k >= 0 else min(k, -EPS)
            q = 1.0 / k / 3.0
            r = 0.5 * rd / k
            disc = q ** 3 + r * r
            ds = math.sqrt(abs(disc))
            if disc >= 0:
                ru = _cbrt(r + ds) + _cbrt(r - ds)
            else:
                s = _cbrt(math.sqrt(r * r + abs(disc)))
                t = math.atan2(ds, r) / 3.0
                ru = s * (-math.cos(t) + math.sqrt(3.0) * math.sin(t))
        else:
            ru = rd
            f = 0.0
            for _ in range(NEWTON_ITERS):
                r2 = ru * ru
                ri = r2
                dist = 1.0 + p[4] * r2
                grad = 1.0 + 3.0 * p[4] * r2
                for j in range(1, nk):
                    ri = r2 * ri
                    dist += p[4 + j] * ri
                    grad += (3 + 2 * j) * p[4 + j] * ri
                f = ru * dist - rd
                ru = ru - f / grad
            valid = abs(f) <= NEWTON_TOL
        sc = ru / max(rd, EPS)
        x, y = mx * sc, my * sc
        n = math.sqrt(x * x + y * y + 1.0)
        return x / n, y / n, 1.0 / n, valid
    if model == KB:
        r = math.hypot(mx, my)
        th = math.sqrt(r)
        f = 0.0
        for _ in range(NEWTON_ITERS):
            t2 = th * th
            ti = t2
            fac = 1.0 + p[4] * t2
            grad = 1.0 + 3.0 * p[4] * t2
            for j in range(1, nk):
                ti = ti * t2
                fac += p[4 + j] * ti
                grad += (3 + 2 * j) * p[4 + j] * ti
            f = th * fac - r
            th = th - f / grad
        s = math.sin(th) / max(r, EPS)
        return s * mx, s * my, math.cos(th), abs(f) <= NEWTON_TOL
    if model == DIVISION:
        r2 = mx * mx + my * my
        ri = r2
        z = 1.0 + p[4] * r2
        for j in range(1, nk):
            ri = ri * r2
            z += p[4 + j] * ri
        n = math.sqrt(r2 + z * z)
        return mx / n, my / n, z / n, True
    xi = p[4]
    r2 = mx * mx + my * my
    tmp = 1.0 + (1.0 - xi * xi) * r2
    fac = (xi + math.sqrt(max(tmp, 0.0))) / (1.0 + r2)
    return fac * mx, fac * my, fac - xi, tmp >= 0


@njit(cache=True, error_model="numpy")
def _jac_bearing(model, nk, p, bx, by, bz, u, v, jb):
    """cam.jac_bearings_wrt_params for one pixel into jb (3, D), then cam.get_optim_jac."""
    d = 4 + nk
    for a in range(3):
        for c in range(d):
            jb[a, c] = 0.0
    fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    if model == PINHOLE:
        iz = 1.0 / max(bz, EPS)
        px, py = bx * iz, by * iz
        xin = 1.0 / math.sqrt(px * px + py * py + 1.0)
        x3 = (px, py, 1.0)
        dbm = np.empty((3, 2))
        for a in range(3):
            dbm[a, 0] = xin * ((1.0 if a == 0 else 0.0) - xin * xin * x3[a] * px)
            dbm[a, 1] = xin * ((1.0 if a == 1 else 0.0) - xin * xin * x3[a] * py)
        ifx, ify = -1.0 / fx, -1.0 / fy
        for a in range(3):
            jb[a, 0] = dbm[a, 0] * px * ifx
            jb[a, 1] = dbm[a, 1] * py * ify
            jb[a, 2] = dbm[a, 0] * ifx
            jb[a, 3] = dbm[a, 1] * ify
        return
    if model == RADIAL:
        iz = 1.0 / max(bz, EPS)
        pux, puy = bx * iz, by * iz
        ru2 = pux * pux + puy * puy
        powers = np.empty(nk)
        ri = ru2
        powers[0] = ru2
        agg1 = 2.0 * p[4]
        agg2 = p[4] * ru2
        for i in range(1, nk):
            agg1 += (2 + 2 * i) * p[4 + i] * ri
            ri = ri * ru2
            powers[i] = ri
            agg2 += p[4 + i] * ri
        a00, a01 = 1.0 + agg2 + agg1 * pux * pux, agg1 * pux * puy
        a10, a11 = agg1 * puy * pux, 1.0 + agg2 + agg1 * puy * puy
        m0 = (a00 * iz, a01 * iz, -(a00 * pux + a01 * puy) * iz)
        m1 = (a10 * iz, a11 * iz, -(a10 * pux + a11 * puy) * iz)
        g0 = (m1[1] * bz - m1[2] * by, m1[2] * bx - m1[0] * bz, m1[0] * by - m1[1] * bx)
        g1 = (by * m0[2] - bz * m0[1], bz * m0[0] - bx * m0[2], bx * m0[1] - by * m0[0])
        den = m0[0] * g0[0] + m0[1] * g0[1] + m0[2] * g0[2]
        inx, iny = -1.0 / max(fx, EPS), -1.0 / max(fy, EPS)
        pdx, pdy = (cx - u) * inx, (cy - v) * iny
        for a in range(3):
            ga, gb = g0[a] / den, g1[a] / den
            jb[a, 0] = ga * pdx * inx
            jb[a, 1] = gb * pdy * iny
            jb[a, 2] = ga * inx
            jb[a, 3] = gb * iny
            for i in range(nk):
                jb[a, 4 + i] = -(ga * pux * powers[i] + gb * puy * powers[i])
        return
    if model == KB:
        ifx, ify = 1.0 / fx, 1.0 / fy
        rr = math.hypot(bx, by)
        th = math.atan2(rr, bz)
        sx, sy = (u - cx) * ifx, (v - cy) * ify
        sir = 1.0 / max(math.hypot(sx, sy), EPS)
        csx, csy = sx * sir, sy * sir
        dbt = (bz * csx, bz * csy, -rr)
        t2 = th * th
        ti = t2
        dsr = 1.0 + 3.0 * p[4] * t2
        dtk = np.empty(nk)
        dtk[0] = t2 * th
        for i in range(1, nk):
            ti = ti * t2
            dsr += (3 + 2 * i) * p[4 + i] * ti
            dtk[i] = ti * th
        for i in range(nk):
            dtk[i] = -dtk[i] / dsr
        dtdr = 1.0 / dsr
        dr_dc = (-ifx * csx, -ify * csy)
        dr_df = (dr_dc[0] * sx, dr_dc[1] * sy)
        cs = (csx, csy)
        s_ = (sx, sy)
        if_ = (ifx, ify)
        dr = (dr_df[0], dr_df[1], dr_dc[0], dr_dc[1])
        for a in range(3):
            for c in range(4):
                jb[a, c] = dbt[a] * dtdr * dr[c]
        for a in range(2):
            for b in range(2):
                dcdc = if_[b] * sir * (cs[a] * cs[b] - (1.0 if a == b else 0.0))
                jb[a, b] += rr * s_[b] * dcdc
                jb[a, 2 + b] += rr * dcdc
        for a in range(3):
            for i in range(nk):
                jb[a, 4 + i] = dbt[a] * dtk[i]
        return
    if model == DIVISION:
        nix, niy = -1.0 / fx, -1.0 / fy
        mx, my = (cx - u) * nix, (cy - v) * niy
        mnx, mny = mx * nix, my * niy
        r2 = mx * mx + my * my
        ri = r2
        mz = 1.0 + p[4] * r2
        dmz = p[4]
        for i in range(1, nk):
            dmz += (i + 1) * p[4 + i] * ri
            ri = ri * r2
            mz += p[4 + i] * ri
        inorm = 1.0 / (r2 + mz * mz)
        sq = math.sqrt(inorm)
        m3 = (mx, my, mz)
        for a in range(3):
            d0 = sq * ((1.0 if a == 0 else 0.0) - inorm * m3[a] * mx)
            d1 = sq * ((1.0 if a == 1 else 0.0) - inorm * m3[a] * my)
            d2 = sq * ((1.0 if a == 2 else 0.0) - inorm * m3[a] * mz)
            jb[a, 0] = d0 * mnx + d2 * dmz * 2.0 * mx * mnx
            jb[a, 1] = d1 * mny + d2 * dmz * 2.0 * my * mny
            jb[a, 2] = d0 * nix + d2 * dmz * 2.0 * mnx
            jb[a, 3] = d1 * niy + d2 * dmz * 2.0 * mny
            pw = r2
            for i in range(nk):
                jb[a, 4 + i] = d2 * pw
                pw = pw * r2
        return
    ifx, ify = 1.0 / fx, 1.0 / fy
    xi = p[4]
    mx, my = (u - cx) * ifx, (v - cy) * ify
    r2 = mx * mx + my * my
    ir2p1 = 1.0 / (r2 + 1.0)
    tmp = math.sqrt(max(1.0 + (1.0 - xi * xi) * r2, 0.0))
    fac = (xi + tmp) * ir2p1
    dfac = ir2p1 * (0.5 * (1.0 - xi * xi) / tmp - fac)
    dcx, dcy = -2.0 * mx * ifx, -2.0 * my * ify
    dfc = (mx * dcx, my * dcy, dcx, dcy)
    m3 = (mx, my, 1.0)
    for a in range(3):
        for c in range(4):
            jb[a, c] = dfac * m3[a] * dfc[c]
    jb[0, 0] += 0.5 * fac * dcx
    jb[1, 1] += 0.5 * fac * dcy
    jb[0, 2] -= fac * ifx
    jb[1, 3] -= fac * ify
    dxi = (tmp - r2 * xi) * ir2p1 / tmp
    jb[0, 4] = mx * dxi
    jb[1, 4] = my * dxi
    jb[2, 4] = dxi - 1.0
    s = 2.0 * math.sqrt(xi)
    for a in range(3):
        jb[a, 4] *= s


@njit(cache=True, error_model="numpy")
def _logmap(x0, x1, x2, y0, y1, y2, g):
    """Unit3.logmap(q, p) and Unit3.jac_logmap_wrt_refvecs(q, p) (into g, 2 x 3): (r0, r1, valid). q = (x0, x1,
    x2) the model's ray, p = (y0, y1, y2) the observed one.
    """
    pole = abs(x1) >= 1.0 - EPS
    e00, e02 = (1.0, 0.0) if pole else (x2, -x0)
    norm = math.sqrt(e00 * e00 + e02 * e02)
    e00, e02 = e00 / norm, e02 / norm
    e10, e11, e12 = x1 * e02, x2 * e00 - x0 * e02, -x1 * e00
    cos_t = x0 * y0 + x1 * y1 + x2 * y2
    npar = cos_t < 1.0 - EPS_PARALLEL
    chord = math.sqrt((x0 - y0) ** 2 + (x1 - y1) ** 2 + (x2 - y2) ** 2)
    t = 2.0 * math.asin(min(max(0.5 * chord, 0.0), 1.0 - EPS))
    by0 = e00 * y0 + e02 * y2
    by1 = e10 * y0 + e11 * y1 + e12 * y2
    sc = t / math.sin(t) if npar else 1.0
    inv_cub = 1.0 / max(e11 * e11 * e11, EPS)
    c = cos_t if npar else 0.0
    cosec2 = 1.0 / (1.0 - c * c)
    tcs = t * math.sqrt(cosec2)
    t1 = cosec2 * (1.0 - c * tcs) if npar else 1.0 / 3.0
    xy = x0 * y0 + x2 * y2
    x012 = x0 * x1 * x2
    s2 = tcs if npar else 1.0
    g[0, 0] = t1 * by0 * (x0 - y0) + s2 * (-x2 * xy * inv_cub)
    g[0, 1] = t1 * by0 * (x1 - y1)
    g[0, 2] = t1 * by0 * (x2 - y2) + s2 * (x0 * xy * inv_cub)
    g[1, 0] = t1 * by1 * (x0 - y0) + s2 * (
        (x0 ** 3 * y1 + x012 * y2 + x0 * x2 * x2 * y1 - x1 * x2 * x2 * y0) * inv_cub)
    g[1, 1] = t1 * by1 * (x1 - y1) + s2 * (-xy / max(e11, EPS))
    g[1, 2] = t1 * by1 * (x2 - y2) + s2 * (
        (-(x0 * x0) * x1 * y2 + x0 * x0 * x2 * y1 + x012 * y0 + x2 ** 3 * y1) * inv_cub)
    return sc * by0, sc * by1, inv_cub < 1.0 / EPS


@njit(cache=True, error_model="numpy", inline="always")
def _point(model, nk, p, x, ray, i, acc, jb, g, jr):
    d = 4 + nk
    u, v = x[i, 0], x[i, 1]
    bx, by, bz, ok = _unproject(model, nk, p, u, v)
    r0, r1, ok2 = _logmap(bx, by, bz, np.float64(ray[i, 0]), np.float64(ray[i, 1]), np.float64(ray[i, 2]), g)
    if not (ok and ok2):
        return
    _jac_bearing(model, nk, p, bx, by, bz, u, v, jb)
    for c in range(d):
        jr[0, c] = g[0, 0] * jb[0, c] + g[0, 1] * jb[1, c] + g[0, 2] * jb[2, c]
        jr[1, c] = g[1, 0] * jb[0, c] + g[1, 1] * jb[1, c] + g[1, 2] * jb[2, c]
    for a in range(d):
        acc[d * d + a] -= jr[0, a] * r0 + jr[1, a] * r1
        for b in range(d):
            acc[a * d + b] += jr[0, a] * jr[0, b] + jr[1, a] * jr[1, b]
    acc[d * d + d] += r0 * r0 + r1 * r1
    acc[d * d + d + 1] += 1.0


@njit(cache=True, error_model="numpy")
def _pass_serial(model, nk, p, x, ray, acc):
    d = 4 + nk
    acc[:] = 0.0
    jb, g, jr = np.empty((3, d)), np.empty((2, 3)), np.empty((2, d))
    for i in range(len(x)):
        _point(model, nk, p, x, ray, i, acc, jb, g, jr)


@njit(cache=True, error_model="numpy", parallel=True)
def _pass_parallel(model, nk, p, x, ray, acc):
    d = 4 + nk
    n = len(x)
    part = np.zeros((N_CHUNKS, len(acc)))
    step = (n + N_CHUNKS - 1) // N_CHUNKS
    for c in prange(N_CHUNKS):
        jb, g, jr = np.empty((3, d)), np.empty((2, 3)), np.empty((2, d))
        for i in range(c * step, min(n, (c + 1) * step)):
            _point(model, nk, p, x, ray, i, part[c], jb, g, jr)
    acc[:] = 0.0
    for c in range(N_CHUNKS):
        acc += part[c]


@njit(cache=True, error_model="numpy")
def _pass(model, nk, p, x, ray, acc):
    """`GaussNewtonCalib.res_and_jac_in_fitted_tangent` + its normal equations over the valid points: acc = (J'J
    (D x D, flat), -J'r (D), sum of squared residuals, valid count).
    """
    if len(x) >= PARALLEL_MIN:
        _pass_parallel(model, nk, p, x, ray, acc)
    else:
        _pass_serial(model, nk, p, x, ray, acc)


@njit(cache=True, error_model="numpy")
def _linear_fit(model, nk, x, ray, p, hold, cx0, cy0, simple=False):
    """cam.fit for one image into p; False if the system is singular / not finite. `hold`: ``c = (cx0, cy0)``
    given (ucm / division: Eq. 11's ``c`` replaced, its fy / fx kept). `simple` (with `hold` only): upstream's
    ``simple_<model>``, one focal for both axes -- no Eq. 11 (ucm / division: ``fy / fx = 1``), and the
    pinhole-type systems with one ``1 / f`` column for both axes, normalized by the largest coordinate of
    either (``amax((-2, -1))``).
    """
    n = len(x)
    d = 4 + nk
    if model == UCM or model == DIVISION:
        ata = np.zeros((3, 3))
        atb = np.zeros(3)
        for i in range(n):
            X, Y = np.float64(ray[i, 0]), np.float64(ray[i, 1])
            row = (Y * x[i, 0], -Y, X)
            t = x[i, 1] * X
            for a in range(3):
                atb[a] += row[a] * t
                for b in range(3):
                    ata[a, b] += row[a] * row[b]
        sol = np.empty(3)
        ok = _solve(ata, atb, sol)
        ar = sol[0]
        cx, cy = sol[1] / ar, sol[2]
        if hold:
            cx, cy = cx0, cy0
        if simple:
            ar, ok = 1.0, True
        if model == DIVISION:
            q = 1 + nk
            a3 = np.zeros((q, q))
            b3 = np.zeros(q)
            row3 = np.empty(q)
            for i in range(n):
                X, Y, Z = np.float64(ray[i, 0]), np.float64(ray[i, 1]), np.float64(ray[i, 2])
                xc, yc = x[i, 0] - cx, x[i, 1] - cy
                rc = math.hypot(xc, yc)
                ra = math.hypot(X, ar * Y)
                rca2 = xc * xc + yc * yc / (ar * ar)
                row3[0] = ra
                row3[1] = ra * rca2
                for j in range(2, q):
                    row3[j] = row3[j - 1] * rca2
                t = rc * Z
                for a in range(q):
                    b3[a] += row3[a] * t
                    for b in range(q):
                        a3[a, b] += row3[a] * row3[b]
            s3 = np.empty(q)
            ok = _solve(a3, b3, s3) and ok
            fx = s3[0]
            p[0], p[1], p[2], p[3] = fx, ar * fx, cx, cy
            fp = fx
            for j in range(nk):
                p[4 + j] = s3[1 + j] * fp
                fp = fp * fx * fx
            return ok and np.isfinite(p[:d]).all()
        a2 = np.zeros((2, 2))
        b2 = np.zeros(2)
        fsum, fcnt = 0.0, 0.0
        for i in range(n):
            X, Y, Z = np.float64(ray[i, 0]), np.float64(ray[i, 1]), np.float64(ray[i, 2])
            rc = math.hypot(x[i, 0] - cx, x[i, 1] - cy)
            ra = math.hypot(X, ar * Y)
            dd = math.sqrt(X * X + Y * Y + Z * Z)
            r_ = (ra, -rc * dd)
            t = rc * Z
            for a in range(2):
                b2[a] += r_[a] * t
                for b in range(2):
                    a2[a, b] += r_[a] * r_[b]
            if ra > EPS and Z > EPS:
                fsum += rc * Z / max(ra, EPS)
                fcnt += 1.0
        s2 = np.empty(2)
        ok = _solve(a2, b2, s2) and ok
        fx, xi = s2[0], s2[1]
        if xi < 0:
            fx, xi = fsum / max(fcnt, 1.0), 0.0
        p[0], p[1], p[2], p[3], p[4] = fx, ar * fx, cx, cy, xi
        return ok and np.isfinite(p[:5]).all()
    ata = np.zeros((d, d))
    atb = np.zeros(d)
    row = np.empty(d)
    if model == KB:
        fac_x, fac_y = 1.0, 1.0
    else:
        fac_x, fac_y = -np.inf, -np.inf
        for i in range(n):
            fac_x, fac_y = max(fac_x, x[i, 0]), max(fac_y, x[i, 1])
        if simple:
            fac_x = fac_y = max(fac_x, fac_y)
    for i in range(n):
        X, Y, Z = np.float64(ray[i, 0]), np.float64(ray[i, 1]), np.float64(ray[i, 2])
        if model == KB:
            rr = math.hypot(X, Y)
            th = math.atan2(rr, Z)
            ox, oy = th * X / max(rr, EPS), th * Y / max(rr, EPS)
            w2 = th * th
        else:
            if not math.hypot(X, Y) < Z * TAN_HALF_FOV:
                continue
            iz = 1.0 / max(Z, EPS)
            ox, oy = X * iz, Y * iz
            w2 = ox * ox + oy * oy
        for axis in range(2):
            o = ox if axis == 0 else oy
            row[:] = 0.0
            if hold:
                row[0 if simple else axis] = (x[i, axis] - (cx0 if axis == 0 else cy0)) / (fac_x if axis == 0 else fac_y)
            else:
                row[axis] = x[i, axis] / (fac_x if axis == 0 else fac_y)
                row[2 + axis] = -1.0
            if model != PINHOLE:
                c_ = -o * w2
                for j in range(nk):
                    row[4 + j] = c_
                    c_ *= w2
            for a in range(d):
                atb[a] += row[a] * o
                for b in range(d):
                    ata[a, b] += row[a] * row[b]
    if hold:
        for j in range(d):
            ata[2, j], ata[3, j], ata[j, 2], ata[j, 3] = 0.0, 0.0, 0.0, 0.0
        ata[2, 2], ata[3, 3], atb[2], atb[3] = 1.0, 1.0, 0.0, 0.0
        if simple:
            for j in range(d):
                ata[1, j], ata[j, 1] = 0.0, 0.0
            ata[1, 1], atb[1] = 1.0, 0.0
    sol = np.empty(d)
    ok = _solve_scaled(ata, atb, sol) if model == KB else _solve(ata, atb, sol)
    if model != KB:
        for j in range(d):
            if not np.isfinite(sol[j]):
                sol[j] = 1.0
    p[0], p[1] = fac_x / sol[0], (fac_x / sol[0] if simple else fac_y / sol[1])
    p[2], p[3] = (cx0, cy0) if hold else (sol[2] * p[0], sol[3] * p[1])
    for j in range(nk):
        p[4 + j] = sol[4 + j]
    return ok and np.isfinite(p[:d]).all()


@njit(cache=True, error_model="numpy")
def _update(model, nk, p, delta):
    """cam.get_optim_update: p + delta (ucm: xi + 2 sqrt(xi) d + d^2)."""
    d = 4 + nk
    if model == UCM:
        for j in range(4):
            p[j] += delta[j]
        xi = p[4]
        p[4] = xi + 2.0 * math.sqrt(xi) * delta[4] + delta[4] * delta[4]
    else:
        for j in range(d):
            p[j] += delta[j]


@njit(cache=True, error_model="numpy")
def _solve_tied(jtj, rhs, delta, simple, hold):
    """The Gauss-Newton system in the free parameters only: `simple` ties fx and fy into one f (upstream's
    simple_* parameterization: the step of f is the step of both, its Jacobian the sum of theirs); `hold`
    drops cx, cy (`fix_cxcy`). The step is expanded back to all parameters.
    """
    d = len(rhs)
    grp = np.empty(d, dtype=np.int64)
    m = 0
    for a in range(d):
        if a == 1 and simple:
            grp[a] = grp[0]
        elif (a == 2 or a == 3) and hold:
            grp[a] = -1
        else:
            grp[a] = m
            m += 1
    a_r = np.zeros((m, m))
    b_r = np.zeros(m)
    for a in range(d):
        ga = grp[a]
        if ga < 0:
            continue
        b_r[ga] += rhs[a]
        for b in range(d):
            if grp[b] >= 0:
                a_r[ga, grp[b]] += jtj[a, b]
    sol = np.empty(m)
    ok = _solve(a_r, b_r, sol)
    for a in range(d):
        delta[a] = 0.0 if grp[a] < 0 else sol[grp[a]]
    return ok


@njit(cache=True, error_model="numpy")
def _solve_held(jtj, rhs, delta):
    """`fix_cxcy`: the system without cx, cy's rows and columns (upstream's `optim_idx`), its step expanded with
    zeros there (`expand_updates`).
    """
    d = len(rhs)
    m = d - 2
    a = np.empty((m, m))
    b = np.empty(m)
    sol = np.empty(m)
    for i in range(m):
        ii = i if i < 2 else i + 2
        b[i] = rhs[ii]
        for j in range(m):
            a[i, j] = jtj[ii, j if j < 2 else j + 2]
    ok = _solve(a, b, sol)
    delta[2], delta[3] = 0.0, 0.0
    for i in range(m):
        delta[i if i < 2 else i + 2] = sol[i]
    return ok


@njit(cache=True, error_model="numpy")
def _calibrate_one(model, nk, x, ray, iters, out, hold, cx0, cy0, simple=False):
    """Calibrator.__call__ for one image (linear fit, Gauss-Newton, keep if better) into out; returns the linear
    fit's success. `hold`: ``c = (cx0, cy0)`` given and kept; `simple`: one focal (upstream's
    ``simple_<model>``), reported as ``fx == fy``.
    """
    d = 4 + nk
    p = np.empty(d)
    ok = _linear_fit(model, nk, x, ray, p, hold, cx0, cy0, simple)
    acc = np.empty(d * d + d + 2)
    _pass(model, nk, p, x, ray, acc)
    cost0 = acc[d * d + d] / acc[d * d + d + 1]
    q = p.copy()
    jtj = np.empty((d, d))
    rhs = np.empty(d)
    delta = np.empty(d)
    for _ in range(iters):
        for a in range(d):
            rhs[a] = acc[d * d + a]
            for b in range(d):
                jtj[a, b] = acc[a * d + b]
        if simple:
            solved = _solve_tied(jtj, rhs, delta, True, hold)
        else:
            solved = _solve_held(jtj, rhs, delta) if hold else _solve(jtj, rhs, delta)
        if not solved:
            delta[:] = 0.0
        for a in range(d):
            if not np.isfinite(delta[a]):
                delta[a] = 0.0
        _update(model, nk, q, delta)
        _pass(model, nk, q, x, ray, acc)
    cost = acc[d * d + d] / acc[d * d + d + 1]
    src = q if cost < cost0 else p
    for j in range(d):
        out[j] = src[j]
    return ok


@njit(cache=True, error_model="numpy")
def _calibrate_batch(model, nk, x, rays, iters, cxcy, hold, simple=False):
    bsz = rays.shape[0]
    out = np.empty((bsz, 4 + nk))
    ok = np.empty(bsz, dtype=np.bool_)
    for b in range(bsz):
        ok[b] = _calibrate_one(model, nk, x, rays[b], iters, out[b], hold, cxcy[b, 0], cxcy[b, 1], simple)
    return out, ok


def held_cxcy(model: int, cam_id: str, cxcy, bsz: int) -> tuple[np.ndarray, bool]:
    """``cxcy`` (B, 2) / (2,) / None -> a (B, 2) float64 array and whether it is held; only every model takes a
    known ``c``.
    """
    if cxcy is None:
        return np.zeros((bsz, 2)), False
    if (isinstance(cxcy, np.ndarray) and cxcy.dtype == np.float64 and cxcy.shape == (bsz, 2)
            and cxcy.flags.c_contiguous):
        return cxcy, True
    c = np.asarray(cxcy.detach().cpu() if hasattr(cxcy, "cpu") else cxcy, dtype=np.float64)
    return np.broadcast_to(c.reshape(-1, 2), (bsz, 2)).copy(), True


def fit_rays(cam_id: str, rays, x, iters: int = 10, cxcy=None, simple: bool = False) -> np.ndarray:
    """AnyCalib's calibrator on (B, N, 3) rays at the pixels ``x`` (N, 2) of the field's frame (float32 or
    float64), exactly ``iters`` Gauss-Newton iterations -> (B, 4 + nk) intrinsics (fx, fy, cx, cy, k...; ucm:
    xi) in that frame. ``cxcy`` (B, 2) or (2,): the principal point given in that frame and held, as
    upstream's ``data["cxcy"]``. ``simple``: upstream's ``simple_<cam_id>`` (one focal, square pixels), with
    ``cxcy`` only; returned as ``fx == fy``.
    """
    model, nk = model_of(cam_id)
    rays = np.ascontiguousarray(rays)
    c, hold = held_cxcy(model, cam_id, cxcy, rays.shape[0])
    if simple and not hold:
        raise ValueError("simple: ported with a held principal point only (cxcy)")
    out, _ = _calibrate_batch(model, nk, np.ascontiguousarray(x, dtype=np.float64), rays,
                              int(iters), c, hold, bool(simple))
    return out
