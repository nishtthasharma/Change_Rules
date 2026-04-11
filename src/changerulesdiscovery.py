import numpy as np
import time

class ChangeRuleDiscovery:

    def __init__(self, differential_functions, computed_bitsets):
        self.Psi = differential_functions
        self.Bitsets = computed_bitsets
        self.delta_cols = list(differential_functions.keys())

    @staticmethod
    def _strictly_contains(outer, inner):
        return (
            outer[0] <= inner[0]
            and outer[1] >= inner[1]
            and outer != inner
        )

    def discover_rules(self, lhs_attr, chi=0.7):
        rules = []
        m = None

        for entry_X in self.Psi[lhs_attr]:
            g_X = entry_X["interval"]
            bitset_X = self.Bitsets[lhs_attr][g_X]

            if m is None:
                m = len(bitset_X)

            max_seg_X = entry_X["max_seg"]
            start_idx = entry_X["start_idx"]

            if max_seg_X == 0:
                continue

            segment_end = start_idx + max_seg_X

            for Y in self.delta_cols:
                if Y == lhs_attr:
                    continue

                qualifying_rules = []
                for entry_Y in self.Psi[Y]:
                    g_Y = entry_Y["interval"]

                    bitset_Y = self.Bitsets[Y][g_Y]
                    
                    # Calculate Confidence
                    overlap_count = np.count_nonzero(bitset_Y[start_idx:segment_end])

                    if overlap_count == 0:
                        continue

                    conf = overlap_count / max_seg_X

                    if conf >= chi:
                        qualifying_rules.append({
                            "lhs_attr": lhs_attr,
                            "lhs_interval": g_X,
                            "rhs_attr": Y,
                            "rhs_interval": g_Y,
                            "confidence": conf,
                            "support_global": np.count_nonzero(bitset_X & bitset_Y) / m,
                            "overlap_count": overlap_count
                        })

                for candidate in qualifying_rules:
                    rhs_interval = candidate["rhs_interval"]
                    is_minimal = not any(
                        self._strictly_contains(rhs_interval, other["rhs_interval"])
                        for other in qualifying_rules
                    )
                    if is_minimal:
                        rules.append(candidate)

        # print(f"\nTotal minimal rules discovered for {lhs_attr}: {len(rules)}")
        return rules
