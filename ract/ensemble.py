import numpy as np
from scipy.special import comb
from sklearn.ensemble import BaggingClassifier
from ract.tree import RecourseTreeClassifier
from ract.utils import find_best_actions, compute_action_indicators

try:
    import gurobipy as gp
    import os, contextlib
    HAS_GRB = True
except:
    HAS_GRB = False


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
    
    
    def _get_thresholds(self):
        thresholds = []
        feature_pointer = [ 0 ]
        for d in range(self.action.n_features):
            thresholds_d = []
            for n_estimator in range(self.n_estimators):
                thresholds_d += self.estimators_[n_estimator].tree_.threshold[self.estimators_[n_estimator].tree_.feature == d].tolist()
            thresholds_d = sorted(list(set(thresholds_d)))
            if len(thresholds_d) > 0:
                thresholds.append(list(zip([d] * len(thresholds_d), thresholds_d)))
            feature_pointer.append(feature_pointer[-1] + len(thresholds_d))
        thresholds = np.concatenate(thresholds, axis=0)
        return thresholds, feature_pointer
    
    def _get_regions_and_labels(self):
        regions = []
        labels = []
        tree_pointer = [ 0 ]
        for estimator in self.estimators_:
            leaves = np.array([ j for j in range(estimator.tree_.node_count) if estimator.tree_.feature[j] < 0])
            if len(leaves) == 0: continue
            regions.append(estimator.regions_[leaves])
            labels.append((estimator.tree_.label[leaves] == self.action.y_target).astype(np.int64))            
            tree_pointer.append(tree_pointer[-1] + len(leaves))
        regions = np.concatenate(regions, axis=0)
        labels = np.concatenate(labels, axis=0)
        return regions, labels, tree_pointer
    
    def _get_milo_model(self, x, A, C, I, labels, max_change_features, confidence):

        n_features = self.action.n_features
        As = [A[A[:, 0] == d] for d in range(n_features)]
        Cs = [C[A[:, 0] == d] for d in range(n_features)]
        J = [len(A_d) for A_d in As]
        lb = [min(A_d[:, 1]) for A_d in As]
        ub = [max(A_d[:, 1]) for A_d in As]
        L = [I_t.shape[0] for I_t in I]

        with contextlib.redirect_stdout(open(os.devnull, 'w')): model = gp.Model()
        def LinSum(Vars): return gp.LinExpr([1]*len(Vars), Vars)        
        def flatten(x): return sum(x, [])

        act = [
            model.addVar(name='act_{:04d}'.format(d), vtype=gp.GRB.CONTINUOUS, lb=lb[d], ub=ub[d]) for d in range(n_features)
        ] 
        pi = [
            [model.addVar(name='pi_{:04d}_{:04d}'.format(d, j), vtype=gp.GRB.BINARY) for j in range(J[d])] for d in range(n_features)
        ] 
        cost = model.addVar(name='cost', vtype=gp.GRB.CONTINUOUS, lb=0)
        phi  = [
            [model.addVar(name='phi_{:04d}_{:04d}'.format(t, l), vtype=gp.GRB.BINARY) for l in range(L[t])] for t in range(self.n_estimators) 
        ] 

        model.setObjective(cost, gp.GRB.MINIMIZE)
        if self.action.cost_type == 'MPS':
            for d in range(n_features):
                if (d in flatten(self.action.feature_categories) and np.min(As[d][:, 1]) < 0) or self.action.feature_constraints[d] == 'F':
                    continue
                model.addConstr(cost - gp.LinExpr(Cs[d], pi[d]) >= 0, name='C_cost_{:04d}'.format(d))
        else:
            model.addConstr(cost - gp.LinExpr(C, flatten(pi)) == 0, name='C_cost')

        for d in range(n_features): 
            model.addConstr(LinSum(pi[d]) == 1, name='C_basic_pi_{:04d}'.format(d))
            model.addConstr(act[d] - gp.LinExpr(As[d][:, 1], pi[d]) == 0, name='C_basic_act_{:04d}'.format(d))

        nonzeros = (A[:, 1] != 0)
        model.addConstr(gp.LinExpr(nonzeros, flatten(pi)) <= max_change_features, name='C_basic_sparsity')

        for i, G in enumerate(self.action.feature_categories): 
            model.addConstr(LinSum([act[d] for d in G]) == 0, name='C_basic_category_{:04d}'.format(i))

        model.addConstr(gp.LinExpr(labels, flatten(phi)) >= self.n_estimators * (confidence + 1e-8), name='C_loss')

        for t in range(self.n_estimators):
            model.addConstr(LinSum(phi[t]) == 1, name='C_forest_leaf_{:04d}'.format(t))
            for l in range(L[t]):
                model.addConstr(n_features * phi[t][l] - gp.LinExpr(I[t][l], flatten(pi)) <= 0, name='C_forest_decision_{:04d}_{:04d}'.format(t, l))
        
        self.variables_ = {}
        self.variables_['act'] = act
        self.variables_['pi'] = pi
        self.variables_['cost'] = cost
        self.variables_['phi'] = phi

        return model
    
    def explain_exact_action(self, x,
                             max_change_features=-1, confidence=-1, time_limit=60, verbose=False):
        
        X = x.reshape(1, -1)

        if confidence < 0:
            confidence = 0.5       
        if max_change_features < 0:
            max_change_features = self.action.n_features

        thresholds, feature_pointer = self._get_thresholds()
        A = self.action._get_action(X, thresholds)
        C = self.action._get_cost(X, A, feature_pointer)
        A = A[0]; C = C[0]; 
        A_ins = np.zeros((A.shape[0] + self.action.n_features, A.shape[1]), dtype=np.float64)
        C_ins = np.zeros(A_ins.shape[0], dtype=np.float64)
        for d in range(self.action.n_features):
            A_ins[(feature_pointer[d]+d), 0] = d
            A_ins[(feature_pointer[d]+d+1):(feature_pointer[d+1]+d+1)] = A[feature_pointer[d]:feature_pointer[d+1]]    
            C_ins[feature_pointer[d]+d+1:feature_pointer[d+1]+d+1] = C[feature_pointer[d]:feature_pointer[d+1]]            
        A = A_ins[C_ins != np.inf]; C = C_ins[C_ins != np.inf]; 
        
        regions, labels, tree_pointer = self._get_regions_and_labels()
        I = []
        for t in range(self.n_estimators):
            regions_t = regions[tree_pointer[t]:tree_pointer[t+1]]
            I_t = compute_action_indicators(x, A, regions_t)
            I.append(I_t)

        model = self._get_milo_model(x, A, C, I, labels, max_change_features, confidence)
        model.params.outputflag = int(verbose)
        model.params.timelimit = time_limit
        model.optimize()
        self.time_ = model.runtime

        As = [A[A[:, 0] == d] for d in range(self.action.n_features)]
        try:
            a = np.array([ np.sum([a * round(p.X) for p, a in zip(pi_d, A_d[:, 1]) ]) for pi_d, A_d in zip(self.variables_['pi'], As) ])
            solved = True
        except AttributeError:
            a = np.zeros(self.action.n_features)
            solved = False

        if solved:
            c = self.variables_['cost'].X
            v = (self.predict(X + a) == self.action.y_target)[0]
            cv = (c <= self.action.cost_budget) * v
            s = np.count_nonzero(a)
        else: 
            c = 0
            v = False
            cv = False
            s = 0

        results = {
            'solved': solved, 
            'y_target': self.action.y_target, 
            'sample': x, 
            'action': a, 
            'counterfactual': x + a, 
            'cost': c, 
            'validity': v, 
            'cost-validity': cv,
            'sparsity': s,
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








