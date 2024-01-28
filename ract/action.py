import numpy as np
from scipy.stats import median_abs_deviation
from scipy.stats import gaussian_kde as kde
from scipy.interpolate import interp1d
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from lingam import DirectLiNGAM
from ract.utils import MIN_VAL, MAX_VAL, compute_candidate_actions, compute_all_actions, is_feasible

FEATURE_TYPES = {
    'C': 0,
    'B': 1,
    'I': 2, 
}
FEATURE_CONSTRAINTS = {
    'N': 0,
    'F': 1,
    'I': 2,
    'D': 3,
}


class Action():
    def __init__(self, X, 
                 y_target=0, cost_type='MPS', cost_budget=0.5, cost_ord=-1, alpha=0.0, 
                 plausibility_type=None, plausibility_constraint=0.9, n_plausibility=10, causal=False, tol=1e-6,
                 feature_names=[], feature_types=[], feature_constraints=[], feature_categories=[],
                 target_name='Outcome', class_names=['Good', 'Bad']):

        self.X = X
        self.n_samples, self.n_features = X.shape
        self.feature_names = feature_names if len(feature_names)==self.n_features else ['x_{}'.format(d) for d in range(self.n_features)]
        self.feature_types = np.array(feature_types) if len(feature_types)==self.n_features else np.array(['C'] * self.n_features)
        self.feature_constraints = np.array(feature_constraints) if len(feature_constraints)==self.n_features else np.array(['N'] * self.n_features)
        self.feature_categories = feature_categories
        self.target_name = target_name
        self.class_names = class_names

        self.y_target = y_target
        self.cost_type = cost_type
        self.cost_budget = cost_budget
        self.cost_ord = cost_ord
        self.alpha = alpha
        self.tol = tol    
        
        if cost_type in ['MPS', 'TLPS']:
            self.weight_ = None
            self.percentile_ = self._get_percentile(X)
            self.precision_, self.cholesky_ = None, None
        elif cost_type in ['mahalanobis', 'DACE']:
            self.precision_, self.cholesky_ = self._get_precision(X)
            self.weight_ = self.cholesky_.sum(axis=1) ** 2
            self.percentile_ = None
        else:
            self.weight_ = self._get_weight(X, cost_type)
            self.percentile_ = None
            self.precision_, self.cholesky_ = None, None
            
        self.plausibility_type = plausibility_type
        self.plausibility_constraint = plausibility_constraint
        self.k_plausibility = n_plausibility
        if plausibility_type == 'lof':
            self.plausibility_ = LocalOutlierFactor(n_neighbors=n_plausibility, novelty=True).fit(X)
        elif plausibility_type == 'if':
            self.plausibility_ = IsolationForest(n_estimators=n_plausibility).fit(X)        
        else:
            self.plausibility_ = None
            
        self.causal = causal
        if causal:
            self.causal_effects_ = self._get_causal_effects(X)
        else:
            self.causal_effects_ = None
            

    def _get_weight(self, X, cost_type):
        weight = np.ones(self.n_features)
        if cost_type == 'MAD':
            weight = (median_abs_deviation(X) + self.tol) ** -1
            weight[self.feature_types == 'B'] = (X[:, self.feature_types == 'B'] * 1.4826).std(axis=0)
        elif(cost_type == 'STD'):
            weight = np.std(X, axis=0) ** -1
            weight[self.feature_types == 'B'] = (X[:, self.feature_types == 'B'] * 1.4826).std(axis=0)
        elif cost_type == 'normalize':
            weight = (X.max(axis=0) - X.min(axis=0)) ** -1
        return weight
    
    def _get_percentile(self, X, l_buff=1e-6, r_buff=1e-6, l_quantile=0.001, r_quantile=0.999, grid_size=100):
        percentile = []
        for d in range(self.n_features):
            if self.feature_constraints[d] == 'F':
                percentile.append(None)
                continue
            kde_estimator = kde(X[:, d])
            grid = np.linspace(np.quantile(X[:, d], l_quantile), np.quantile(X[:, d], r_quantile), grid_size)
            pdf = kde_estimator(grid)
            cdf_raw = np.cumsum(pdf)
            total = cdf_raw[-1] + l_buff + r_buff
            cdf = (l_buff + cdf_raw) / total
            p_d = interp1d(x=grid, y=cdf, copy=False, fill_value=(l_buff, 1.0 - r_buff), bounds_error=False, assume_sorted=False)
            percentile.append(p_d)
        return percentile

    def _get_precision(self, X):
        est = EmpiricalCovariance(store_precision=True, assume_centered=False).fit(X)
        covariance = est.covariance_
        if np.linalg.matrix_rank(covariance) != X.shape[1]:
            covariance += 1e-6 * np.eye(X.shape[1])
        precision = np.linalg.inv(covariance)
        cholesky = np.linalg.cholesky(precision)
        return precision, cholesky

    def _get_causal_effects(self, X, filter=0.01, max_hop=10):
        adjacency_matrix = DirectLiNGAM().fit(X).adjacency_matrix_
        adjacency_matrix[abs(adjacency_matrix) < filter] = 0.0
        causal_effects = np.zeros_like(adjacency_matrix)
        tmp = np.eye(X.shape[1])
        n_max_hop = min(max_hop, X.shape[1])
        for _ in range(n_max_hop):
            tmp = np.dot(tmp, adjacency_matrix)
            causal_effects = causal_effects + tmp
        return causal_effects

    def _get_action(self, X, thresholds):
        feature_types = np.array([FEATURE_TYPES[feature_type] for feature_type in self.feature_types], dtype=np.int64)
        return compute_candidate_actions(X, thresholds, feature_types)

    def _get_cost(self, X, A, feature_pointer):
        C = np.zeros((A.shape[0], A.shape[1]), dtype=np.float64)
        for d in range(X.shape[1]):
            if self.feature_constraints[d] == 'F':
                C[:, feature_pointer[d]:feature_pointer[d+1]] = np.inf
            else:
                A_d = A[:, feature_pointer[d]:feature_pointer[d+1], 1]
                if self.cost_type in ['MPS', 'TLPS']:
                    q_d = self.percentile_[d]
                    q_0 = q_d(X[:, d])
                    X_cf = np.tile(X[:, d].reshape(-1, 1), A_d.shape[1]) + A_d
                    if self.cost_type == 'MPS':
                        C[:, feature_pointer[d]:feature_pointer[d+1]] = abs(np.tile(q_0.reshape(-1, 1), A_d.shape[1]) - q_d(X_cf))
                    else:
                        C[:, feature_pointer[d]:feature_pointer[d+1]] = abs(np.log2((1 - np.tile(q_0.reshape(-1, 1), A_d.shape[1])) / (1 - q_d(X_cf))))
                elif self.cost_type in ['mahalanobis', 'DACE']:
                    C[:, feature_pointer[d]:feature_pointer[d+1]] = (self.cholesky_[d].sum() * abs(A_d)) ** 2
                else:
                    C[:, feature_pointer[d]:feature_pointer[d+1]] = self.weight_[d] * abs(A_d)
                if self.feature_constraints[d] == 'I':
                    C[:, feature_pointer[d]:feature_pointer[d+1]][A_d<0] = np.inf
                elif self.feature_constraints[d] == 'D':
                    C[:, feature_pointer[d]:feature_pointer[d+1]][A_d>0] = np.inf
        return C

    def enumerate_actions(self, X, regions, cost_type=None, max_change_features=-1, spacer=True, l_quantile=0.01, r_quantile=0.99):
        feature_types = np.array([FEATURE_TYPES[feature_type] for feature_type in self.feature_types], dtype=np.int64)
        feature_constraints = np.array([FEATURE_CONSTRAINTS[feature_constraint] for feature_constraint in self.feature_constraints], dtype=np.int64)

        if (cost_type is not None) and (self.cost_type != cost_type):
            if cost_type in ['MPS', 'TLPS']:
                self.percentile_ = self._get_percentile(self.X)
            elif cost_type in ['mahalanobis', 'DACE']:
                self.precision_, self.cholesky_ = self._get_precision(self.X)
                self.weight_ = self.cholesky_.sum(axis=1) ** 2
            else:
                self.weight_ = self._get_weight(self.X, cost_type)       
            self.cost_type = cost_type     

        cost_category = np.zeros(self.n_features, dtype=np.float64)
        for cats in self.feature_categories:
            if self.cost_type == 'MPS':
                cost_category[cats] += np.array([100.0 if self.percentile_[d] is None 
                                                 else abs(self.percentile_[d](1) - self.percentile_[d](0)) for d in cats])                
            elif self.cost_type == 'TLPS':
                cost_category[cats] += np.array([100.0 if self.percentile_[d] is None 
                                                 else abs(np.log2((1-self.percentile_[d](1)) / (1-self.percentile_[d](0)))) for d in cats])                
            else:
                cost_category[cats] += self.weight_[cats]

        if spacer:
            space = (np.quantile(self.X, r_quantile, axis=0) - np.quantile(self.X, l_quantile, axis=0)) / (self.n_samples * (r_quantile - l_quantile)) 
            space = np.array([0.0 if self.feature_constraints[d] == 'F' or self.feature_types[d] == 'B' 
                              else (int(space[d] + 1) if self.feature_types[d] == 'I' else space[d]) for d in range(self.n_features)], dtype=np.float64)
        else:
            space = np.zeros(self.n_features, dtype=np.float64)
       
        null_region = np.zeros((1, self.n_features, 2), dtype=np.float64)
        null_region[:, :, 0] = MIN_VAL
        null_region[:, :, 1] = MAX_VAL
        regions = np.concatenate([regions, null_region])
       
        A_all = compute_all_actions(X, regions, feature_types)
        A = np.array([self._feasify(X[i], A_all[i], regions, cost_category, space) for i in range(X.shape[0])], dtype=np.float64)
        F = is_feasible(A, feature_constraints, max_change_features)
        C = np.array([self._cost(X[i], A[i]) for i in range(X.shape[0])])

        if self.plausibility_ is not None:
            P = -1 * self.plausibility_.score_samples(np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0)).reshape(F.shape) <= self.plausibility_constraint
            F = P * F

        return A, C, F

    def _feasify(self, x, A, regions, cost_category, space):
        for cats in self.feature_categories:
            i = A[:, cats].sum(axis=1)
            if (i == 1).sum() > 0:
                d = cats[np.where(x[cats] == 1)[0][0]]
                A[i == 1, d] = -1.0
            if (i == -1).sum() > 0:
                for l in np.arange(regions.shape[0])[i == -1]:
                    r = regions[l]
                    cats_feasible = [cats[j] for j, r_d in enumerate(r[cats]) if r_d[1] != 0.5]
                    d_feasible = cats_feasible[np.argmin([c_d for r_d, c_d in zip(r[cats], cost_category[cats]) if r_d[1] != 0.5])]
                    A[l, d_feasible] = 1.0
        for d in range(A.shape[1]):
            if space[d] == 0:
                continue
            A[A[:, d] > 0, d] = A[A[:, d] > 0, d] + space[d]
            A[A[:, d] < 0, d] = A[A[:, d] < 0, d] - space[d]                    
        return A        
    
    def _cost(self, x, A):
        if self.cost_type in ['MPS', 'TLPS']:
            C = np.zeros(A.shape[0])
            for d in range(A.shape[1]):
                q_d = self.percentile_[d]
                if q_d is None:
                    continue
                if self.cost_type == 'MPS':
                    C = np.maximum(C, abs(q_d(x[d]) - q_d(x[d] + A[:, d])))
                else:
                    C += abs( np.log2( (1 - q_d(x[d] + A[:, d])) / (1 - q_d(x[d])) ) )
        elif self.cost_type in ['mahalanobis', 'DACE']:
            C = np.diag(np.dot(np.dot(A, self.precision_), A.T))
        else:
            if self.cost_ord == -1:
                C = np.max(abs(A) * self.weight_, axis=1)
            else:
                C = (abs(A) ** self.cost_ord).dot(self.weight_) 
        return C

    def do_intervention(self, A):
        if self.causal_effects_ is None:
            self.causal_effects_ = self._get_causal_effects(self.X)
        return self.causal_effects_.dot(A.T).T

    def print_action(self, x, a):
        for d in range(self.n_features):
            if a[d] == 0: continue
            if self.feature_types[d] == 'B':
                print('- {}: {} -> {}'.format(self.feature_names[d], bool(x[d]), bool(x[d] + a[d])))
            elif self.feature_types[d] == 'I':
                print('- {}: {} -> {} ({:+})'.format(self.feature_names[d], int(x[d]), int(x[d] + a[d]), int(a[d])))
            else:
                print('- {:.4}: {:.4} -> {:.4} ({:+})'.format(self.feature_names[d], x[d], x[d] + a[d], a[d]))