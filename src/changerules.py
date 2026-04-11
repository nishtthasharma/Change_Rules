import itertools
import numpy as np

class ChangeRuleDiscovery:
    def __init__(self, diffset_obj, differential_functions):
        self.diffset = diffset_obj
        self.dfs = differential_functions
        self.delta_cols = diffset_obj.delta_cols
    
    def cover(self, target_col, interval_idx):
        mask = ~self.diffset.diffset_for_function(target_col, interval_idx)
        non_cover_indices = np.where(mask)[0]

        candidates = []
        for col in self.delta_cols:
            if col == target_col:
                continue
            for idx in range(len(self.dfs[col])):
                candidates.append(frozenset([(col, idx)]))

        final_candidates = set(candidates)
        for tup_idx in non_cover_indices:
            decoded = self.diffset.decode_diffset(self.diffset.compute_diffset().iloc[tup_idx])
            to_remove = set()
            to_add = set()

            for cand in final_candidates:
                # If candidate does NOT intersect with tuple, it's invalid for this diff-set
                if not any((col == c and idx == decoded[c]) for (c, idx) in cand):
                    to_remove.add(cand)
                    # Expand candidate by adding other λ(A) in the tuple except target
                    new_cand = set(cand)
                    for c, idx in decoded.items():
                        if c != target_col and (c, idx) not in new_cand:
                            expanded = frozenset(new_cand | {(c, idx)})
                            to_add.add(expanded)
            final_candidates -= to_remove
            final_candidates |= to_add

        return final_candidates