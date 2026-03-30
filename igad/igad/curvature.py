import numpy as np
from typing import Callable, Optional

def fisher_metric(log_partition, theta, eps=1e-4):
    theta = np.asarray(theta, dtype=np.float64)
    d = theta.shape[0]
    g = np.zeros((d, d))
    A = log_partition
    h = eps
    f0 = A(theta)
    for i in range(d):
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        g[i, i] = (A(tp) - 2.0 * f0 + A(tm)) / (h * h)
        for j in range(i + 1, d):
            tpp = theta.copy(); tpp[i] += h; tpp[j] += h
            tpm = theta.copy(); tpm[i] += h; tpm[j] -= h
            tmp = theta.copy(); tmp[i] -= h; tmp[j] += h
            tmm = theta.copy(); tmm[i] -= h; tmm[j] -= h
            g[i, j] = (A(tpp) - A(tpm) - A(tmp) + A(tmm)) / (4.0 * h * h)
            g[j, i] = g[i, j]
    return g

def third_cumulant_tensor(log_partition, theta, eps=1e-3, **kwargs):
    theta = np.asarray(theta, dtype=np.float64)
    d = theta.shape[0]
    T = np.zeros((d, d, d))
    A = log_partition
    h = eps
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                if i == j == k:
                    tp2 = theta.copy(); tp2[i] += 2*h
                    tp1 = theta.copy(); tp1[i] += h
                    tm1 = theta.copy(); tm1[i] -= h
                    tm2 = theta.copy(); tm2[i] -= 2*h
                    val = (A(tp2) - 2*A(tp1) + 2*A(tm1) - A(tm2)) / (2*h*h*h)
                elif i == j:
                    def _g_ii(t, _i=i, _h=h, _A=A):
                        tp = t.copy(); tp[_i] += _h
                        tm = t.copy(); tm[_i] -= _h
                        return (_A(tp) - 2.0*_A(t) + _A(tm)) / (_h*_h)
                    tkp = theta.copy(); tkp[k] += h
                    tkm = theta.copy(); tkm[k] -= h
                    val = (_g_ii(tkp) - _g_ii(tkm)) / (2.0*h)
                elif j == k:
                    def _g_jj(t, _j=j, _h=h, _A=A):
                        tp = t.copy(); tp[_j] += _h
                        tm = t.copy(); tm[_j] -= _h
                        return (_A(tp) - 2.0*_A(t) + _A(tm)) / (_h*_h)
                    tip = theta.copy(); tip[i] += h
                    tim = theta.copy(); tim[i] -= h
                    val = (_g_jj(tip) - _g_jj(tim)) / (2.0*h)
                else:
                    val = 0.0
                    for si in (+1, -1):
                        for sj in (+1, -1):
                            for sk in (+1, -1):
                                t = theta.copy()
                                t[i] += si*h; t[j] += sj*h; t[k] += sk*h
                                val += si*sj*sk*A(t)
                    val /= (8.0*h*h*h)
                for a, b, c in {(i,j,k),(i,k,j),(j,i,k),(j,k,i),(k,i,j),(k,j,i)}:
                    T[a, b, c] = val
    return T

def scalar_curvature(log_partition, theta, g=None, T=None):
    theta = np.asarray(theta, dtype=np.float64)
    if g is None:
        g = fisher_metric(log_partition, theta)
    if T is None:
        T = third_cumulant_tensor(log_partition, theta)
    g_inv = np.linalg.inv(g)
    S = np.einsum("ab,abm->m", g_inv, T)
    S_norm_sq = np.einsum("mn,m,n->", g_inv, S, S)
    T_norm_sq = np.einsum("ia,jb,kc,ijk,abc->", g_inv, g_inv, g_inv, T, T)
    return 0.25 * (S_norm_sq - T_norm_sq)
