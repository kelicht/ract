import numpy as np
from scipy.special import comb
from sklearn.ensemble import BaggingClassifier
from ract.tree import RecourseTreeClassifier
from ract.utils import find_best_actions



class BaseRecourseEnsemble():
    def __init__(self):
        pass

    @property
    def feature_importances_(self):
        all_importances = np.zeros(self.n_features_in_)
        for estimator in self.estimators_:
            all_importances += estimator.feature_importances_
        return all_importances / all_importances.sum()

    def _get_regions_ft(self, confidence, max_search):
        regions = []
        for estimator in self.estimators_:
            leaves = np.array([ j for j in range(estimator.tree_.node_count) if estimator.tree_.feature[j] < 0 and estimator.tree_.label[j] == self.action.y_target])
            if len(leaves) == 0: continue
            regions.append(estimator.regions_[leaves])
        if len(regions) == 0: 
            return []
        regions = np.concatenate(regions, axis=0)
        if regions.shape[0] > max_search:
            regions = regions[:max_search]
        return regions

    def _get_regions_lr(self, confidence, max_search):
        y = self.predict_proba(self.action.X)[:, self.action.y_target]
        X = self.action.X[y >= confidence]
        y = y[y >= confidence]
        y_order = np.argsort(y)[::-1]
        X, y = X[y_order[:max_search]], y[y_order[:max_search]]
        L = np.array([ estimator.tree_.apply(X) for estimator in self.estimators_ ], dtype=np.int64)
        R = np.array([ self.estimators_[t].regions_[L[t]] for t in range(L.shape[0]) ], dtype=np.float64)
        regions = np.zeros((X.shape[0], X.shape[1], 2))
        regions[:, :, 0] = R[:, :, :, 0].max(axis=0)
        regions[:, :, 1] = R[:, :, :, 1].min(axis=0) 
        return regions

    def explain_action(self, X, 
                       cost_type=None, max_change_features=-1, confidence=-1, 
                       search_strategy='ft', max_search=1000, spacer=True):

        if X.shape[0] == 0:
            return {
                'y_target': self.action.y_target, 
                'sample': X, 
                'action': np.zeros_like(X), 
                'counterfactual': X, 
                'cost': np.zeros(1), 
                'validity': np.ones(1, dtype=np.bool_), 
                'cost-validity': np.ones(1, dtype=np.bool_),
                'valid-cost': np.zeros(1), 
                'sparsity': np.zeros(1), 
                'valid-sparsity': np.zeros(1), 
                }

        if confidence < 0:
            confidence = 0.5

        if search_strategy == 'ft':
            regions = self._get_regions_ft(confidence, max_search)
        elif search_strategy == 'lr':
            spacer = False
            regions = self._get_regions_lr(confidence, max_search)
            
        if len(regions) == 0:
            return {
                'y_target': self.action.y_target, 
                'sample': X, 
                'action': np.zeros_like(X), 
                'counterfactual': X, 
                'cost': np.zeros(X.shape[0]), 
                'validity': np.zeros(X.shape[0], dtype=np.bool_), 
                'cost-validity': np.zeros(X.shape[0], dtype=np.bool_),
                'valid-cost': np.zeros(X.shape[0]), 
                'sparsity': np.zeros(X.shape[0]), 
                'valid-sparsity': np.zeros(X.shape[0]), 
                }            

        A, C, F = self.action.enumerate_actions(X, regions, cost_type, max_change_features, spacer=spacer)
        if self.action.causal:
            A_causal = A + self.action.do_intervention(np.concatenate(A, axis=0)).reshape(A.shape)
            F = F * (self.predict_proba(np.repeat(X, A.shape[1], axis=0) + np.concatenate(A_causal, axis=0))[:, self.action.y_target] >= confidence).reshape(F.shape)
        else:
            F = F * (self.predict_proba(np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0))[:, self.action.y_target] >= confidence).reshape(F.shape)

        CA = find_best_actions(X, A, F, C)
        A_opt, C_opt = CA[:, 1:], CA[:, 0]
        if self.action.causal:
            X_cf = X + A_opt + self.action.do_intervention(A_opt)
        else:
            X_cf = X + A_opt
        V = (self.predict(X_cf) == self.action.y_target)
        CV = V * (C_opt <= self.action.cost_budget)
        VC = C_opt[V]
        S = np.count_nonzero(A_opt, axis=1)
        VS = S[V]
        
        results = {
            'y_target': self.action.y_target, 
            'sample': X, 
            'action': A_opt, 
            'counterfactual': X_cf, 
            'cost': C_opt, 
            'validity': V, 
            'cost-validity': CV,
            'valid-cost': VC, 
            'sparsity': S,
            'valid-sparsity': VS,
        }
        return results
    


class RecourseForestClassifier(BaggingClassifier, BaseRecourseEnsemble):
    def __init__(self, action, 
                 max_depth=3, min_sample_leaf=1, max_features='sqrt', n_thresholds=-1, relabeling=False, delta=0.2, feature_masking=False,
                 n_estimators=10, max_samples=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, 
                 warm_start=False, n_jobs=-1, random_state=None, verbose=0):
        self.action = action
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.relabeling = relabeling
        self.delta = delta
        self.feature_masking = feature_masking

        if feature_masking:
            self.action.alpha = 0.0

        super().__init__(estimator=RecourseTreeClassifier(action, max_depth, min_sample_leaf, max_features, n_thresholds, relabeling, delta, feature_masking), 
                         n_estimators=n_estimators, max_samples=max_samples, max_features=1.0, 
                         bootstrap=bootstrap, bootstrap_features=bootstrap_features, oob_score=oob_score, 
                         warm_start=warm_start, n_jobs=n_jobs, random_state=random_state, verbose=verbose)








