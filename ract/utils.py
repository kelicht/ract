import numpy as np
import numba
import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt


MIN_VAL = - 1e+8
MAX_VAL =   1e+8


def export_mermaid(graph):
  graphbytes = graph.encode("ascii")
  base64_bytes = base64.b64encode(graphbytes)
  base64_string = base64_bytes.decode("ascii")
  display(Image(url="https://mermaid.ink/img/" + base64_string))

@numba.njit("int64[:](float64[:, :], int64[:], float64[:], int64[:], int64[:], int64)", parallel=True, cache=True)
def apply(X, feature, threshold, children_left, children_right, j0):
    J = np.zeros(X.shape[0], dtype=np.int64) + j0
    for i in numba.prange(X.shape[0]):
        while children_left[J[i]] >= 0 and children_right[J[i]] >= 0:
            if X[i, feature[J[i]]] <= threshold[J[i]]:
                J[i] = children_left[J[i]]
            else:
                J[i] = children_right[J[i]]
    return J

@numba.njit("float64[:, :, :](int64[:], float64[:], int64[:], int64[:], int64, int64)", cache=True)
def region(feature, threshold, children_left, children_right, n_features_in, j0):
    R = np.zeros((j0+1, n_features_in, 2), dtype=np.float64)
    R[:, :, 0] = MIN_VAL
    R[:, :, 1] = MAX_VAL
    j = j0
    while j != 0:
        if feature[j] < 0:
            j = j- 1
            continue
        else:
            R[children_left[j]] = R[j]
            R[children_right[j]] = R[j]
            R[children_left[j], feature[j], 1] = threshold[j]
            R[children_right[j], feature[j], 0] = threshold[j]
            j = j- 1
    return R

@numba.njit("float64[:, :](float64[:, :], int64[:], int64[:, :], int64[:, :], boolean[:], float64[:, :], int64[:], int64, float64, int64[:], int64[:], int64[:], boolean[:, :])", cache=True)
def compute_loss(X, y, sort_idx, sort_idx_all, is_in, thresholds, feature_mask, y_target, alpha, losses, is_invalid, is_reach, is_flip):
    N = X.shape[0]
    n_node = np.zeros(2)
    for i in sort_idx[:, 0]:
        n_node[y[i]] = n_node[y[i]] + 1
    loss_out = losses[~is_in].sum()
    N_node = n_node.sum()
    M_b = (is_invalid * is_reach).sum()
    M_o = (is_invalid * (1 - is_reach)).sum()
    loss = np.zeros((thresholds.shape[0], 6), dtype=np.float64)
    loss[:, 2] = MAX_VAL
    loss[:, 3] = MAX_VAL
    d_prev = -1
    for k, (d, t) in enumerate(thresholds):
        d = int(d)
        if not feature_mask[d]: 
            continue
        if d != d_prev:
            n_left = np.zeros(2)
            n_right = np.array([n for n in n_node])
            N_l = 0
            N_r = N_node
            i = 0
        while (i < sort_idx.shape[0]) and (X[sort_idx[i, d], d] <= t):
            n_left[y[sort_idx[i, d]]] = n_left[y[sort_idx[i, d]]] + 1
            n_right[y[sort_idx[i, d]]] = n_right[y[sort_idx[i, d]]] - 1
            N_l = N_l + 1
            N_r = N_r - 1
            i = i + 1
        if d != d_prev:
            M_l = 0
            M_r = M_b
            j_l = 0
            j_r = 0          
        while (j_l < N) and ((X[sort_idx_all[j_l, d], d] <= t) or is_flip[sort_idx_all[j_l, d], k]):
            M_l = M_l + is_invalid[sort_idx_all[j_l, d]] * is_reach[sort_idx_all[j_l, d]]
            j_l = j_l + 1
        while (j_r < N) and (X[sort_idx_all[j_r, d], d] <= t) and (not is_flip[sort_idx_all[j_r, d], k]):
            M_r = M_r - is_invalid[sort_idx_all[j_r, d]] * is_reach[sort_idx_all[j_r, d]]
            j_r = j_r + 1
        loss[k, 0] = d
        loss[k, 1] = t
        for y_left in [0, 1]:
            for y_right in [0, 1]:
                loss_k = loss_out + n_left[1 - y_left] + n_right[1 - y_right]
                if alpha > 0:
                    if y_left == y_target:
                        if y_right == y_target:
                            recourse_loss_k = M_o
                        else:
                            recourse_loss_k = M_b - M_l + M_o
                    else:
                        if y_right == y_target:
                            recourse_loss_k = M_b - M_r + M_o
                        else:
                            recourse_loss_k = M_b + M_o
                else:
                    recourse_loss_k = 0
                if loss_k + alpha * recourse_loss_k < loss[k, 2] + alpha * loss[k, 3]:
                    loss[k, 2] = loss_k
                    loss[k, 3] = recourse_loss_k
                    loss[k, 4] = y_left
                    loss[k, 5] = y_right
        d_prev = d
    return loss

@numba.njit("void(float64[:, :], int64, float64, int64[:, :], int64[:, :], int64[:, :])", parallel=True, cache=True)
def split_sort_idx(X, split_dim, split_th, sort_idx, sort_idx_left, sort_idx_right):
    for d in numba.prange(sort_idx.shape[1]):
        l, r = 0, 0
        for i in sort_idx[:, d]:
            if X[i, split_dim] <= split_th:
                sort_idx_left[l, d] = int(i)
                l = l + 1
            else:
                sort_idx_right[r, d] = int(i)
                r = r + 1

@numba.njit("boolean[:](float64, boolean[:], int64[:, :], float64[:])", cache=True)
def minimum_set_cover(delta, initial_solution, is_covered, loss_grad):
    is_selected = initial_solution.copy()
    while is_selected.sum() < is_selected.shape[0]:
        coverage = (is_covered[is_selected].sum(axis=0) > 0).mean()
        if coverage >= 1 - delta: break
        j_opt = -1
        grad_opt = 0.0
        leaves_candidate = np.where(~is_selected)[0]
        for j in leaves_candidate:
            is_selected_j = is_selected.copy()
            is_selected_j[j] = True
            coverage_j = (is_covered[is_selected_j].sum(axis=0) > 0).mean()
            grad = ((coverage_j - coverage) / loss_grad[j])
            if grad > grad_opt:
                j_opt = j
                grad_opt = grad
        if j_opt == -1:
            break
        else:
            is_selected[j_opt] = True
    return is_selected

@numba.njit("float64[:, :, :](float64[:, :], float64[:, :], int64[:])", parallel=True, cache=True)
def compute_candidate_actions(X, thresholds, feature_types):
    A = np.zeros((X.shape[0], thresholds.shape[0], 2), dtype=np.float64)
    for i in numba.prange(X.shape[0]):
        for j in numba.prange(thresholds.shape[0]):
            d, b = thresholds[j]
            A[i, j, 0] = d
            d = int(d)
            if X[i, d] <= b:
                if feature_types[d] == 1:
                    A[i, j, 1] = 1.0
                elif feature_types[d] == 2:
                    A[i, j, 1] = int(b) - X[i, d] + 1.0
                else:
                    A[i, j, 1] = b - X[i, d] + 1e-6
            else:
                if feature_types[d] == 1:
                    A[i, j, 1] = - 1.0
                elif feature_types[d] == 2:
                    A[i, j, 1] = int(b) - X[i, d]
                else:
                    A[i, j, 1] = b - X[i, d]                    
    return A

@numba.njit("float64[:, :, :](float64[:, :], float64[:, :, :], int64[:])", parallel=True, cache=True)
def compute_all_actions(X, regions, feature_types):
    A = np.zeros((X.shape[0], regions.shape[0], X.shape[1]), dtype=np.float64)
    for i in numba.prange(X.shape[0]):
        for l in numba.prange(regions.shape[0]):
            for d in numba.prange(X.shape[1]):
                if X[i, d] <= regions[l][d][0]:
                    if feature_types[d] == 1:
                        A[i, l, d] = 1.0
                    elif feature_types[d] == 2:
                        A[i, l, d] = int(regions[l][d][0]) - X[i, d] + 1.0
                    else:
                        A[i, l, d] = regions[l][d][0] - X[i, d] + 1e-8
                elif X[i, d] <= regions[l][d][1]:
                    A[i, l, d] = 0.0
                else:
                    if feature_types[d] == 1:
                        A[i, l, d] = - 1.0
                    elif feature_types[d] == 2:
                        A[i, l, d] = int(regions[l][d][1]) - X[i, d]
                    else:
                        A[i, l, d] = regions[l][d][1] - X[i, d]
    return A

@numba.njit("boolean[:, :](float64[:, :, :], int64[:], int64)", parallel=True, cache=True)
def is_feasible(A, feature_constraints, max_change_features):
    F = np.ones((A.shape[0], A.shape[1]), dtype=np.bool_)
    is_fix = ((feature_constraints == 1).sum() > 0)
    is_inc = ((feature_constraints == 2).sum() > 0)
    is_dec = ((feature_constraints == 3).sum() > 0)
    for i in numba.prange(A.shape[0]):
        if is_fix:
            F[i] = F[i] * (np.count_nonzero(A[i][:, feature_constraints == 1], axis=1) == 0)
        if is_inc:
            F[i] = F[i] * (np.count_nonzero(np.clip(A[i][:, feature_constraints == 2], None, 0.0), axis=1) == 0)
        if is_dec:
            F[i] = F[i] * (np.count_nonzero(np.clip(A[i][:, feature_constraints == 3], 0.0, None), axis=1) == 0)
        if max_change_features > 0: 
            F[i] = F[i] * (np.count_nonzero(A[i], axis=1) <= max_change_features)
    return F

@numba.njit("float64[:, :](float64[:, :], float64[:, :, :], boolean[:, :], float64[:, :])", parallel=True, cache=True)
def find_best_actions(X, A, F, C):
    CA = np.zeros((X.shape[0], 1 + X.shape[1]), dtype=np.float64)
    C_opt = np.ones(X.shape[0], dtype=np.float64) * MAX_VAL
    for i in numba.prange(X.shape[0]):
        if F[i].sum() == 0: continue
        for l in range(F.shape[1]):
            if F[i, l] and (C[i, l] < C_opt[i]):
                CA[i, 0] = C[i, l]
                CA[i, 1:] = A[i, l]
                C_opt[i] = C[i, l]
    return CA