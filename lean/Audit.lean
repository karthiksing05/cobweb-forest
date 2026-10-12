import CobwebForest
-- Every paper statement formalised in Lean; `#print axioms` must show only
-- propext, Classical.choice, Quot.sound (no sorryAx).
-- Discrete cores
#print axioms CobwebForest.validity_nonneg
#print axioms CobwebForest.condExp_const_of_validity_zero
#print axioms CobwebForest.condExp_blocked
#print axioms CobwebForest.conditional_validity_blocked
#print axioms CobwebForest.Chain.fanin_one
#print axioms CobwebForest.ka_tree_count
#print axioms CobwebForest.sign_worlds_local_law
#print axioms CobwebForest.local_rule_same
#print axioms CobwebForest.sign_worlds_differ
#print axioms CobwebForest.parity_fibre_balanced
#print axioms CobwebForest.merged_prediction
#print axioms CobwebForest.merged_weights
#print axioms CobwebForest.leaf_sum_eq
#print axioms CobwebForest.leaf_mixture_eq_nw
-- Representation
#print axioms CobwebForest.univ_tree
#print axioms CobwebForest.univ_tree_counts
#print axioms CobwebForest.grid_rate
#print axioms CobwebForest.Net.stable
#print axioms CobwebForest.Net.stable_of_margins
#print axioms CobwebForest.Net.univ_forest
#print axioms CobwebForest.Net.compose
#print axioms CobwebForest.Net.compose_grid
#print axioms CobwebForest.separation_a
#print axioms CobwebForest.bump_mem_B
#print axioms CobwebForest.separation_b
#print axioms CobwebForest.Net.numTrees_le
#print axioms CobwebForest.cobweb_ka
#print axioms CobwebForest.Net.numTrees_kaNet
-- Learning
#print axioms CobwebForest.learned_tree_eq_nw
#print axioms CobwebForest.learned_trees_inherit
#print axioms CobwebForest.gauss_ratio
#print axioms CobwebForest.gauss_far_perturbed
#print axioms CobwebForest.wavg_perturb
#print axioms CobwebForest.step2_bound
#print axioms CobwebForest.condExp_tower
#print axioms CobwebForest.learn_rate
#print axioms CobwebForest.condExp_eq_of_factor
#print axioms CobwebForest.funnel_xy
#print axioms CobwebForest.suffclass_alg
#print axioms CobwebForest.suffclass
