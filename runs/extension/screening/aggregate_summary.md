# Extension Experiment Summary

## always_adapt

| severity | reward mean | reward SEM | waypoints | success | adapt rate | confidence | delta vs frozen |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 1981.01 | 24.67 | 0.81 | 0.01 | 1.00 | 0.84 | 0.46% |
| mild | 251.91 | 63.01 | 0.53 | 0.00 | 0.35 | 0.78 | 25.20% |
| moderate | 75.29 | 11.42 | 0.25 | 0.00 | 0.15 | 0.68 | -24.06% |
| severe | 35.39 | 4.36 | 0.17 | 0.00 | 0.07 | 0.63 | 20.90% |

## current_default

| severity | reward mean | reward SEM | waypoints | success | adapt rate | confidence | delta vs frozen |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 1968.34 | 5.06 | 0.87 | 0.00 | 0.04 | 0.78 | -0.18% |
| mild | 250.23 | 66.90 | 0.51 | 0.00 | 0.04 | 0.78 | 24.60% |
| moderate | 73.78 | 11.03 | 0.25 | 0.00 | 0.04 | 0.68 | -25.65% |
| severe | 35.46 | 4.37 | 0.17 | 0.00 | 0.04 | 0.62 | 20.97% |

## frozen

| severity | reward mean | reward SEM | waypoints | success | adapt rate | confidence | delta vs frozen |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 1971.88 | 0.00 | 0.88 | 0.00 | 0.00 | n/a | n/a |
| mild | 214.72 | 24.09 | 0.43 | 0.04 | 0.00 | n/a | n/a |
| moderate | 109.71 | 16.34 | 0.28 | 0.01 | 0.00 | n/a | n/a |
| severe | 33.44 | 9.93 | 0.08 | 0.00 | 0.00 | n/a | n/a |

## improved_ppo

| severity | reward mean | reward SEM | waypoints | success | adapt rate | confidence | delta vs frozen |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 1964.77 | 9.03 | 0.81 | 0.00 | 0.04 | 0.80 | -0.36% |
| mild | 262.65 | 73.32 | 0.55 | 0.00 | 0.04 | 0.76 | 30.99% |
| moderate | 75.83 | 12.27 | 0.24 | 0.00 | 0.04 | 0.69 | -23.28% |
| severe | 35.34 | 4.18 | 0.17 | 0.00 | 0.04 | 0.62 | 20.91% |

## improved_ppo_moderate

| severity | reward mean | reward SEM | waypoints | success | adapt rate | confidence | delta vs frozen |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 1909.56 | 18.54 | 0.93 | 0.00 | 0.04 | 0.79 | -3.16% |
| mild | 249.43 | 70.10 | 0.55 | 0.00 | 0.04 | 0.76 | 24.39% |
| moderate | 75.98 | 11.90 | 0.27 | 0.00 | 0.04 | 0.70 | -23.24% |
| severe | 35.48 | 4.40 | 0.17 | 0.00 | 0.04 | 0.63 | 21.01% |

## reward_weighted_rescue

| severity | reward mean | reward SEM | waypoints | success | adapt rate | confidence | delta vs frozen |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 1990.78 | 10.64 | 0.88 | 0.00 | 0.04 | 0.84 | 0.96% |
| mild | 254.64 | 67.83 | 0.52 | 0.00 | 0.04 | 0.79 | 26.85% |
| moderate | 82.16 | 8.03 | 0.25 | 0.00 | 0.04 | 0.73 | -18.89% |
| severe | 35.40 | 4.40 | 0.17 | 0.00 | 0.04 | 0.69 | 20.74% |

## Tuned Method Selection

Selected: `improved_ppo`

No candidate satisfied all constraints; chose highest mean surprise reward.

