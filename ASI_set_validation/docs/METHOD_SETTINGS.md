# Fixed numerical and selection settings

## Local search

Absolute Spearman screening: 0.20. Candidates are tested in descending marginal-association order. GCM pruning threshold: 0.05. Maximum conditioning order: 3 in primary validation, with 2 and 4 evaluated on the matched sensitivity sample. The first separating set removes the candidate. Outcome and treatment searches use the same parent queue and visited/child exclusions. The treatment search does not include outcome as a candidate. Treatment-side dictionary entries take precedence for repeated targets. No conflict-driven graph repair is performed.

## GCM conditional means

Five shared shuffled folds, seed 2026. Conditioning inputs are standardized. Standardized response variables are clipped to ±6 and rescaled. Cubic splines use seven knots, no bias column, and linear extrapolation. Spline features are standardized inside each fitting fold. RidgeCV uses eight penalties logarithmically spaced from 10^-4 to 10. A multi-output cache reuses predictions for repeated conditioning sets. With residual product u, the statistic is sqrt(n) mean(u) / sd(u), with population-variance normalization and a two-sided normal reference. This is the first-order GCM, not a high-order main-screening replacement.

## ANM-HSIC direction decision

Each pair is median-centered and scaled by IQR/1.349, with a sample-standard-deviation fallback. Both directions share five folds. Fold-specific cubic splines use clip(round(n_fit^0.25)+3, 5, 12) knots and ridge penalty 0.01. Normalized centered RBF-HSIC uses median-distance bandwidths and the same at-most-500 sample subset in each direction (seed 2026+11). The score is backward residual dependence minus forward residual dependence. The direction margin is 0.005; it is not a p-value.

Orientation is withheld when both raw HSIC < 0.01 and |Spearman| < 0.10. It is also withheld when the maximum nonlinear R² gain over two directional linear ridge fits is ≤ 0.03 and the four normality-test p-values (both variables and both linear residuals) exceed 0.05. Linear ridge penalty is 10^-6. Raw-pair HSIC uses seed 2026+7. Undecided pairs remain upstream candidates. The direction module requires at least 40 observations and four unique values per variable; the 1,500-observation simulations satisfy these limits.

## Local CI safeguard

For each unordered retained-neighbor pair (A,B) of target V, standardize A and B, clip at ±6, and form their powers 1, 2, and 3. Standardize each power. The marginal diagnostic p0 is min(1,9×minimum p-value) over the nine covariance-product moment tests. The conditional diagnostic p1 is first-order out-of-fold GCM given V. If p0 > 0.10 and p1 < 0.01/max(1,M), where M is the number of retained-neighbor pairs, both neighbors are protected from removal as children. This is a local retention rule, not an additional global graph-learning stage or graph-level type-I-error guarantee.

## Final adjustment-set construction

Construct the back-door graph by deleting treatment's outgoing edges. Graph-defined roots are nodes without an internal parent in the observed common-ancestor set of treatment and outcome in that graph. The candidate domain is the original input graph's treatment/outcome ancestor union, restricted to covariates and excluding treatment descendants. Intersect that domain with roots and their descendants to obtain the root closure.

Enumerate subsets in increasing cardinality, including the empty set. Check d-separation by ancestral moralization. At the first nonempty admissible family, minimize summed directed distance to outcome in the back-door graph; an absent directed path has infinite distance. Break remaining ties by fixed variable identifier. Return an explicit failure for cycles or an exhausted search. This rule does not select by estimation error and does not claim variance optimality.

A source-candidate dependence diagnostic is retained for reproducible CI-call accounting, but does not determine the graph-defined roots or the selected adjustment set. Counts include its conditional tests, screening tests, and singleton local-protection tests; marginal tests are not counted as CI calls.

## Primary validation and matched sensitivity

Primary base seed: 73000419 + 7919r for repeats r=0,...,9. Sensitivity base seed: 2026 + 7919r for r=0,...,4. Test seed adds 10^9. The generator adds 1000×case_id + 37×mechanism_index, with the mechanism order fixed in the JSON configuration. Each run has 1,500 training and 1,500 independent test observations. Structural coefficients are fixed in `data_generators.py`.

Primary comparison arms are estimated-graph GCI, true-graph GCI, all covariates, and no adjustment. The matched sensitivity table evaluates the GCI output at each conditioning order on the same 80 datasets. These sensitivity datasets are not the independent primary validation sample. All settings were fixed before generating the primary validation data.
