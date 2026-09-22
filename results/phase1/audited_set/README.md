# Phase 1 audited set

The output of the Phase 1 audit: the 11,896 designs, out of the 27,720 in the
design space.

| File | Contents |
| --- | --- |
| `accepted_v3.npy` | the indices of the admitted designs |
| `f_min_v3.npy` | the portfolio safety score of every design in the grid |
| `audit_result.json` | the audit statistics |

A design is admitted when its score reaches `tau = 0.9523809523809523`. The audit
passed at that threshold with `N = 2117` paired trials and `k = 0` violations.

The episodes it was decided on are published as the `phase1_audit` dataset.
