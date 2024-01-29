import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from ract.utils import MIN_VAL, MAX_VAL, export_mermaid, apply, region, compute_loss, split_sort_idx, minimum_set_cover, find_best_actions


class Tree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.feature = []
        self.threshold = []
        self.value = []
        self.n_node_samples = []
        self.children_left = []
        self.children_right = []
        self.label = []
        self.is_reachable = []
        self.node_count = 0

    def update(self, feature, threshold, value, n_node_samples, children_left, children_right, label, is_reachable):
        self.feature.append(feature)
        self.threshold.append(threshold)
        self.value.append(value)
        self.n_node_samples.append(n_node_samples)
        self.children_left.append(children_left)
        self.children_right.append(children_right)
        self.label.append(label)
        self.is_reachable.append(is_reachable)
        return self

    def compile(self):
        self.feature = np.array(self.feature)
        self.threshold = np.array(self.threshold)
        self.value = np.array(self.value)
        self.n_node_samples = np.array(self.n_node_samples)
        self.children_left = np.array(self.children_left)
        self.children_right = np.array(self.children_right)
        self.label = np.array(self.label)
        self.is_reachable = np.array(self.is_reachable)
        self.node_count = len(self.feature)
        return self

    def apply(self, X):
        J = apply(X, self.feature, self.threshold, self.children_left, self.children_right, self.node_count-1)
        return J

    def region(self, n_features_in):
        R = region(self.feature, self.threshold, self.children_left, self.children_right, n_features_in, self.node_count-1)
        return R


class RecourseTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, action, 
                 max_depth=3, min_sample_leaf=1, max_features=None, n_thresholds=-1, relabeling=False, delta=0.2, feature_masking=False):

        self.action = action
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.relabeling = relabeling
        self.delta = delta
        self.feature_masking = feature_masking

    def print_tree(self):
        def rec(j, h):
            if self.tree_.feature[j] < 0:
                y = self.tree_.label[j]
                print('\t' * h + '- predict: {} ({:.1%})'.format(self.action.class_names[y], self.tree_.value[j, y]/self.tree_.n_node_samples[j]))
            else:
                d, b = self.tree_.feature[j], self.tree_.threshold[j]
                if self.action.feature_types[d] == 'B':
                    feature_name = self.action.feature_names[d].split(':')
                    if len(feature_name) > 1:
                        print('\t' * h + '- If {} is {}:'.format(feature_name[0], feature_name[1]))
                    else:
                        print('\t' * h + '- If {}:'.format(self.action.feature_names[d]))
                    rec(self.tree_.children_right[j], h+1)
                    print('\t' * h + '- Else:')
                    rec(self.tree_.children_left[j], h+1)
                else:
                    if self.action.feature_types[d] == 'I':
                        print('\t' * h + '- If {} <= {}:'.format(self.action.feature_names[d], int(b)))
                    else:
                        print('\t' * h + '- If {} <= {:.4}:'.format(self.action.feature_names[d], b))
                    rec(self.tree_.children_left[j], h+1)
                    print('\t' * h + '- Else:')
                    rec(self.tree_.children_right[j], h+1)
        rec(self.tree_.node_count-1, 0)

    def to_mermaid(self, top_down=False):
            
        s = """graph {}; """.format('TD' if top_down else 'LR')

        for n in range(self.tree_.node_count-1, -1, -1):
            d, b = self.tree_.feature[n], self.tree_.threshold[n]    
            if d < 0:
                y = self.tree_.label[n]
                s += """N{}[["{}: <b>{}</b>""".format(n, self.action.target_name, self.action.class_names[y])
                s += """ ({:.1%})"]]; """.format(self.tree_.value[n, y]/self.tree_.n_node_samples[n])
            else:
                if self.action.feature_types[d] == 'B':
                    if ':' in self.action.feature_names[d]:
                        prv, nxt = self.action.feature_names[d].split(':')
                        s += """N{}([{} = {}]); """.format(n, prv, nxt)
                    else:
                        s += """N{}([{}]); """.format(n, self.action.feature_names[d])                        
                elif self.action.feature_types[d] == 'I':
                    s += """N{}([{} <= {}]); """.format(n, self.action.feature_names[d], int(b))
                else:
                    s += """N{}([{} <= {:.4}]); """.format(n, self.action.feature_names[d], b)
        
        for n in range(self.tree_.node_count-1, -1, -1):
            n_left, n_right = self.tree_.children_left[n], self.tree_.children_right[n]
            if n_left >= 0 and n_right >= 0:
                if self.action.feature_types[self.tree_.feature[n]] == 'B':
                    s += """N{} -- True --> N{}; """.format(n, n_right)
                    s += """N{} -. False .-> N{}; """.format(n, n_left)
                else:
                    s += """N{} -- True --> N{}; """.format(n, n_left)
                    s += """N{} -. False .-> N{}; """.format(n, n_right)

        return s

    def draw_tree(self, top_down=False):
        export_mermaid(self.to_mermaid(top_down))

    def fit(self, X, y):
        self.tree_ = Tree(self.max_depth)
        self.n_samples_, self.n_features_in_ = X.shape
        self.n_outputs_ = 1

        self.actionable_features_ = np.where(self.action.feature_constraints != 'F')[0]
        self.n_actionable_features_ = len(self.actionable_features_)

        if self.max_features == 'sqrt':
            self.max_features_ = max(1, int(np.sqrt(self.n_actionable_features_))) if self.feature_masking else max(1, int(np.sqrt(self.n_features_in_)))
        else:
            self.max_features_ = self.n_actionable_features_ if self.feature_masking else self.n_features_in_

        self.classes_ = np.unique(y)
        self.feature_importances_ = np.zeros(self.n_features_in_)

        thresholds = []
        feature_pointer = [ 0 ]
        for d in range(self.n_features_in_):
            xd = np.unique(X[:, d])
            xd = (xd[:-1] + (xd[1:] - xd[:-1]) / 2)
            if xd.size > 1: xd = xd[:-1]
            if (self.n_thresholds > 0) and (xd.size > self.n_thresholds):
                xd = xd[np.linspace(0, xd.size-1, self.n_thresholds, dtype=int)]
            thresholds.append(np.stack([np.array([d]*xd.size), xd], axis=1))
            feature_pointer.append(xd.size + feature_pointer[-1])
        thresholds = np.concatenate(thresholds, axis=0)

        y_counts = np.eye(2, dtype=int)[y]
        if y_counts[1-self.action.y_target].sum() < y_counts[self.action.y_target].sum() + self.action.alpha * self.n_samples_:
            y_root = self.action.y_target
            obj_root = y_counts[1-self.action.y_target].sum()
        else:
            y_root = 1 - self.action.y_target    
            obj_root = y_counts[self.action.y_target].sum() + self.action.alpha * self.n_samples_

        losses = (y != y_root).astype(np.int64)
        is_reach = np.ones(self.n_samples_, dtype=np.int64)
        n_valid_leaves = np.ones(self.n_samples_, dtype=np.int64) * int(y_root == self.action.y_target)

        A = self.action._get_action(X, thresholds)
        C = self.action._get_cost(X, A, feature_pointer)
        is_flip = (C <= self.action.cost_budget)

        sort_idx = np.stack([np.argsort(X[:, d]) for d in range(X.shape[1])], axis=1)
        is_in = np.ones(self.n_samples_, dtype=np.bool_)
        _, _, _, _ = self._fit_recursive(X, y, y_counts, y_root, obj_root, sort_idx, sort_idx, is_in, thresholds, self.max_depth, losses, is_reach, n_valid_leaves, is_flip)

        self.tree_ = self.tree_.compile()
        if self.feature_importances_.sum() > 0: self.feature_importances_ /= self.feature_importances_.sum()
        self.regions_ = self.tree_.region(self.n_features_in_)
        
        if self.relabeling:
            self = self.relabel(self.delta)
        return self

    def _fit_recursive(self, X, y, y_counts, label, obj_prev, sort_idx, sort_idx_all, is_in, thresholds, depth, 
                       losses, is_reach, n_valid_leaves, is_flip):        
        value = np.sum(y_counts[sort_idx[:, 0], :], axis=0)
        n_node_samples = np.sum(value)

        if (n_node_samples <= self.min_sample_leaf) or (depth == 0) or (np.sum(value > 0) == 1):
            self.tree_ = self.tree_.update(-2, -2.0, value, n_node_samples, -1, -1, label, is_reach)
            node_idx = len(self.tree_.feature) - 1            
            return node_idx, obj_prev, losses, n_valid_leaves

        feature_mask = np.zeros(self.n_features_in_, dtype=np.int64)
        if self.feature_masking:
            features_sampled = self.actionable_features_[np.random.choice(self.n_actionable_features_, self.max_features_, replace=False)]
        else:
            features_sampled = np.random.choice(self.n_features_in_, self.max_features_, replace=False)
        feature_mask[features_sampled] = 1

        is_invalid = ((n_valid_leaves - (label == self.action.y_target) * is_reach) == 0).astype(np.int64)
        result = compute_loss(X, y, sort_idx, sort_idx_all, is_in, thresholds, feature_mask, 
                              self.action.y_target, self.action.alpha, losses, is_invalid, is_reach, is_flip)

        j = np.argmin(result[:, 2] + self.action.alpha * result[:, 3])
        obj = result[j, 2] + self.action.alpha * result[j, 3]

        feature, threshold = int(result[j, 0]), result[j, 1]
        idx = (X[sort_idx[:, feature], feature] <= threshold)
        if (np.sum(idx) == 0) or (np.sum(~idx) == 0):
            self.tree_ = self.tree_.update(-2, -2.0, value, n_node_samples, -1, -1, label, is_reach)
            node_idx = len(self.tree_.feature) - 1            
            return node_idx, obj_prev, losses, n_valid_leaves

        label_left, label_right = int(result[j, 4]), int(result[j, 5])
        is_in_left = is_in * (X[:, feature] <= threshold)
        is_in_right = is_in * (X[:, feature] > threshold)
        sort_idx_left = np.zeros((np.sum(idx), sort_idx.shape[1]), dtype=int)
        sort_idx_right = np.zeros((np.sum(~idx), sort_idx.shape[1]), dtype=int)
        split_sort_idx(X, feature, threshold, sort_idx, sort_idx_left, sort_idx_right)

        losses[is_in_left] = (y[is_in_left] != label_left)
        losses[is_in_right] = (y[is_in_right] != label_right)
        is_reach_left = is_reach * np.maximum(is_in_left.astype(np.int64), is_flip[:, j])
        is_reach_right = is_reach * np.maximum(is_in_right.astype(np.int64), is_flip[:, j])
        n_valid_leaves = n_valid_leaves - (label == self.action.y_target) * is_reach + (label_left == self.action.y_target) * is_reach_left + (label_right == self.action.y_target) * is_reach_right
        
        children_left, obj, losses, n_valid_leaves = self._fit_recursive(X, y, y_counts, label_left, obj, sort_idx_left, sort_idx_all, is_in_left, thresholds, depth-1, 
                                                                         losses, is_reach_left, n_valid_leaves, is_flip)

        children_right, obj, losses, n_valid_leaves = self._fit_recursive(X, y, y_counts, label_right, obj, sort_idx_right, sort_idx_all, is_in_right, thresholds, depth-1, 
                                                                          losses, is_reach_right, n_valid_leaves, is_flip)
        
        self.feature_importances_[feature] += 1
        self.tree_ = self.tree_.update(feature, threshold, value, n_node_samples, children_left, children_right, label, is_reach)
        node_idx = len(self.tree_.feature) - 1

        return node_idx, obj, losses, n_valid_leaves

    def predict(self, X):
        J = self.tree_.apply(X)
        return self.tree_.label[J]

    def predict_proba(self, X):
        y_pred = self.predict(X)
        y_proba = np.eye(2, dtype=np.float64)[y_pred]
        return y_proba

    def relabel(self, delta):
        leaves = np.array([ j for j in range(self.tree_.node_count) if self.tree_.feature[j] < 0 ])
        is_covered = self.tree_.is_reachable[leaves]
        loss_grad = np.array([(self.tree_.n_node_samples[j] - self.tree_.value[j, self.action.y_target]) / self.n_samples_ for j in leaves], dtype=np.float64)
        
        initial_solution = np.array([self.tree_.label[j] == self.action.y_target for j in leaves], dtype=np.bool_)
        is_selected = minimum_set_cover(delta, initial_solution, is_covered, loss_grad)
        self.tree_.label[leaves[is_selected]] = self.action.y_target
        
        return self

    def explain_action(self, X, cost_type=None, max_change_features=-1):
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

        leaves = np.array([ j for j in range(self.tree_.node_count) if (self.tree_.feature[j] < 0) and self.tree_.label[j] == self.action.y_target ])

        A, C, F = self.action.enumerate_actions(X, self.regions_[leaves], cost_type, max_change_features)
        if self.action.causal:
            A_causal = A + self.action.do_intervention(np.concatenate(A, axis=0)).reshape(A.shape)
            F = F * (self.predict(np.repeat(X, A.shape[1], axis=0) + np.concatenate(A_causal, axis=0)) == self.action.y_target).reshape(F.shape)
        else:
            F = F * (self.predict(np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0)) == self.action.y_target).reshape(F.shape)

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


