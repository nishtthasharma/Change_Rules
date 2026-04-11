import numpy as np
class DiffSet:
    def __init__(self, augmented_df, differential_functions):
        self.df = augmented_df.reset_index(drop=True)
        self.dfs = differential_functions
        self.delta_cols = [c for c in self.df.columns if c.startswith("delta_")]

    def tuple_to_diffset(self, row):

        result = []
        for col in self.delta_cols:
            val = row[col]
            intervals = self.dfs[col]
            idx = 0
            for j, (l, u) in enumerate(intervals):
                if l <= val < u:
                    idx = j
                    break

            result.append(idx)

        return result
    
    def compute_diffset(self):
        print("\nComputing Diff-Sets")
        diffsets = []
        for _, row in self.df.iterrows():
            diffsets.append(self.tuple_to_diffset(row))

        return diffsets
    
    def diffset_for_function(self, col, interval_idx):

        l, u = self.dfs[col][interval_idx]
        mask = (self.df[col] >= l) & (self.df[col] < u)

        return mask