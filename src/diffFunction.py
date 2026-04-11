import numpy as np
import time
import math

class DifferentialFunction:

    def __init__(self, delta_df):
        self.df = delta_df.reset_index(drop=True)
        self.delta_cols = [c for c in self.df.columns if c.startswith("delta_")]

    def longest_consecutive_segment(self, bitset, tau=None):
        true_count = int(np.count_nonzero(bitset))

        if tau is not None and true_count < tau:
            return 0, 0

        if not np.any(bitset):
            return 0, 0

        if tau is not None:
            max_run = 0
            current_run = 0
            start_idx = 0
            temp_start = 0
            remaining_true = true_count

            for i, val in enumerate(bitset):
                if val:
                    if current_run == 0:
                        temp_start = i
                    current_run += 1
                    remaining_true -= 1

                    if current_run > max_run:
                        max_run = current_run
                        start_idx = temp_start

                    if max_run >= tau:
                        return max_run, start_idx
                else:
                    current_run = 0

                # Even if every remaining True value extended the best possible run,
                # this candidate cannot reach tau anymore.
                if max_run + remaining_true < tau:
                    return max_run, start_idx

            return max_run, start_idx

        padded = np.concatenate(([False], bitset, [False])).astype(np.int8)
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        lengths = ends - starts

        best_idx = int(np.argmax(lengths))
        return int(lengths[best_idx]), int(starts[best_idx])

    def discover(self, theta=0.1, priority_col=None, stop_if_missing=False):
        Psi = {}
        Bitsets = {}
        m = len(self.df)
        # print("m:", m)

        tau = math.ceil(theta * m) 
        print(f"Minimum interval length required (tau): {tau}")

        ordered_cols = list(self.delta_cols)
        if priority_col in ordered_cols:
            ordered_cols.remove(priority_col)
            ordered_cols.insert(0, priority_col)

        for col in ordered_cols:
            startTime = time.perf_counter()
            print(f"\nDiscovering intervals for {col}")

            values = self.df[col].values
            unique_vals_sorted = np.sort(np.unique(values))
            if unique_vals_sorted.size < 2:
                Psi[col] = []
                Bitsets[col] = {}
                print(f"{col}: 0 intervals found")
                continue

            min_spacing = np.std(values) / 10 if len(values) > 0 else 0
            if min_spacing > 0:
                keep_mask = np.concatenate((
                    [True],
                    np.diff(unique_vals_sorted) > min_spacing
                ))
                unique_vals = unique_vals_sorted[keep_mask]
            else:
                unique_vals = unique_vals_sorted

            if unique_vals[-1] != unique_vals_sorted[-1]:
                unique_vals = np.append(unique_vals, unique_vals_sorted[-1])
            # print(f"Min value:{min(values)}, unique:{min(unique_vals)}")
            # print(f"Max value:{max(values)}, unique:{max(unique_vals)}")

            psi_A = []
            Bitsets[col] = {}
            minimal_lower_bounds = set()
            candidate_count = 0
            skipped_same_lower_bound = 0

            # ---- Initialize Level 0 ----
            current_level = []
            for i in range(len(unique_vals) - 1):
                l, u = unique_vals[i], unique_vals[i + 1]
                interval = (l, u)
                bitset = (values >= l) & (values <= u)
                
                max_run, start_idx = self.longest_consecutive_segment(bitset, tau)
                
                if max_run >= tau:
                    candidate_count += 1
                    candidate = {
                        "attribute": col,
                        "interval": interval,
                        "support": max_run / m,
                        "max_seg": max_run,
                        "start_idx": start_idx
                    }
                    if l not in minimal_lower_bounds:
                        psi_A.append(candidate)
                        Bitsets[col][interval] = bitset
                        minimal_lower_bounds.add(l)

                current_level.append({
                    "l": l,
                    "u": u,
                    "bitset": bitset
                })

            level = 1
            while len(current_level) > 1:
                next_level = []

                for i in range(len(current_level) - 1):
                    left = current_level[i]
                    right = current_level[i + 1]

                    l, u = left["l"], right["u"]
                    interval = (l, u)

                    if l in minimal_lower_bounds:
                        skipped_same_lower_bound += 1
                        continue

                    new_bitset = left["bitset"] | right["bitset"]

                    max_run, start_idx = self.longest_consecutive_segment(new_bitset, tau)

                    if max_run >= tau:
                        candidate_count += 1
                        if l not in minimal_lower_bounds:
                            candidate = {
                                "attribute": col, 
                                "interval": interval,
                                "support": max_run / m, 
                                "max_seg": max_run, 
                                "start_idx": start_idx
                            }
                            psi_A.append(candidate)
                            Bitsets[col][interval] = new_bitset
                            minimal_lower_bounds.add(l)

                    next_level.append({
                        "l": l,
                        "u": u,
                        "bitset": new_bitset,
                        "max_run": max_run
                    })

                current_level = next_level

            # print(f"{col}: {candidate_count} candidate intervals before minimality")
            # print(f"{col}: {skipped_same_lower_bound} super-interval candidates skipped after reaching support")
            Psi[col] = psi_A
            print(f"{col}: {len(psi_A)} intervals found after minimality")
            # for i in psi_A:
            #     print(f"{i['interval']}, Max Run: {i['max_seg']}, Starts at: {i['start_idx']}")
            endTime = time.perf_counter()
            print(f"Total time for {col}: {endTime - startTime:.4f} seconds")

            if stop_if_missing and col == priority_col and not psi_A:
                print(f"No differential functions found for {priority_col}; skipping remaining attributes.")
                return Psi, Bitsets

        return Psi, Bitsets
