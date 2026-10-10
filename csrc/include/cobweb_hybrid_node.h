#ifndef COBWEB_HYBRID_NODE_H
#define COBWEB_HYBRID_NODE_H

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>
#include <Eigen/Dense>

#include "cobweb_discrete_types.h"

class CobwebHybridTree;

// A Cobweb concept that carries BOTH a discrete attribute-value table (the
// CobwebDiscreteNode representation) and a fixed-size continuous block (the
// CobwebContinuousNode representation: Welford running mean / sum of squared
// deviations, i.e. a diagonal Gaussian per concept).
//
// Both blocks are scored with the same information-theoretic partition
// utility used by the two single-type trees:
//
//     PU = (1/|C|) * [ H(parent) - sum_c P(c) H(c) ]
//
// where H(node) = sum_attr H_discrete(attr) + cont_weight * sum_dim H_gauss(dim).
// H_discrete is the closed-form Dirichlet-smoothed categorical entropy of the
// discrete tree, H_gauss is the differential entropy 0.5 * log(2 pi e var) of
// a diagonal Gaussian whose variance is regularised by ``prior_var`` (the
// continuous analog of ``alpha``).  With ``size == 0`` the node degenerates to
// a plain discrete node; with an empty discrete instance it degenerates to a
// plain continuous node.
class CobwebHybridNode {
    public:
        CobwebHybridTree *tree;
        CobwebHybridNode *parent;
        std::vector<CobwebHybridNode *> children;

        // persistent concept identifier (precomputed on creation)
        std::string concept_id;
        // forest iteration: process-unique integer id, assigned at creation and
        // never reused, so a node can be passed to another tree as a discrete
        // value (its label).  A fringe split keeps the old leaf's uid.
        long long uid;

        COUNT_TYPE count;

        // --- discrete block (mirrors CobwebDiscreteNode) ---
        ATTR_COUNT_TYPE a_count;
        ATTR_COUNT_TYPE sum_n_logn;
        AV_COUNT_TYPE av_count;

        // --- continuous block (mirrors CobwebContinuousNode) ---
        // Welford statistics: empirical variance = sum_sq / count.
        Eigen::VectorXf mean;
        Eigen::VectorXf sum_sq;

        explicit CobwebHybridNode(int size);
        // Deep copy of ``other`` and its subtree (used for fringe splits).
        CobwebHybridNode(CobwebHybridNode *other);
        ~CobwebHybridNode();

        // --- count maintenance ---
        void increment_counts(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont);
        void update_counts_from_node(CobwebHybridNode *other);

        // --- discrete entropies (closed form, per attribute) ---
        double attr_ratio(ATTR_TYPE attr);
        double entropy_attr(ATTR_TYPE attr);
        double entropy_attr_insert(ATTR_TYPE attr, const INSTANCE_TYPE &disc);
        double entropy_attr_merge(ATTR_TYPE attr, CobwebHybridNode *other,
                const INSTANCE_TYPE &disc);
        double entropy_discrete();
        double entropy_discrete_insert(const INSTANCE_TYPE &disc);
        double entropy_discrete_merge(CobwebHybridNode *other, const INSTANCE_TYPE &disc);

        // --- continuous entropies (diagonal Gaussian, closed form) ---
        // Model variance: sum_sq / count + prior_var (prior_var alone when empty).
        Eigen::ArrayXf model_variance(double eval_prior_var = -1.0) const;
        Eigen::ArrayXf variance_insert(const Eigen::VectorXf &cont) const;
        Eigen::ArrayXf variance_merge(CobwebHybridNode *other, const Eigen::VectorXf &cont) const;
        double entropy_continuous();
        double entropy_continuous_insert(const Eigen::VectorXf &cont);
        double entropy_continuous_merge(CobwebHybridNode *other, const Eigen::VectorXf &cont);
        double entropy_continuous_new(const Eigen::VectorXf &cont);

        // --- totals: discrete + continuous ---
        double entropy();
        double entropy_insert(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont);
        double entropy_merge(CobwebHybridNode *other, const INSTANCE_TYPE &disc,
                const Eigen::VectorXf &cont);

        // --- partition utility for the four Cobweb operations ---
        double partition_utility();
        double pu_for_insert(CobwebHybridNode *child, const INSTANCE_TYPE &disc,
                const Eigen::VectorXf &cont);
        double pu_for_new_child(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont);
        double pu_for_merge(CobwebHybridNode *best1, CobwebHybridNode *best2,
                const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont);
        double pu_for_split(CobwebHybridNode *best);
        std::tuple<double, CobwebHybridNode *, CobwebHybridNode *>
        two_best_children(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont);
        OPERATION_TYPE get_best_operation(const INSTANCE_TYPE &disc,
                const Eigen::VectorXf &cont, CobwebHybridNode *best1,
                CobwebHybridNode *best2, double best1_pu);

        bool is_exact_match(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont);

        // --- probabilities ---
        // An empty ``cont`` (size 0) on a tree with size > 0 means "continuous
        // value missing": the Gaussian term is skipped.  Likewise attributes
        // absent from ``disc`` are simply not scored.
        double log_prob_instance(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                double eval_alpha = -1.0, double eval_prior_var = -1.0);
        double log_prob_class_given_instance(const INSTANCE_TYPE &disc,
                const Eigen::VectorXf &cont);
        std::vector<double> log_prob_children_given_instance(const INSTANCE_TYPE &disc,
                const Eigen::VectorXf &cont);
        INSTANCE_TYPE predict_probs();
        double prob(ATTR_TYPE attr, VALUE_TYPE val);

        // --- basic level (closed-form expected PMI against the root) ---
        double expected_pmi_discrete(double eval_alpha = -1.0);
        double expected_pmi_continuous(double eval_prior_var = -1.0);
        double expected_pmi(double eval_alpha = -1.0, double eval_prior_var = -1.0);
        CobwebHybridNode *get_basic(double eval_alpha = -1.0, double eval_prior_var = -1.0);

        // --- misc ---
        std::size_t _hash();
        std::string concept_hash();
        int depth();
        int num_concepts();
        bool is_parent(CobwebHybridNode *other);
};

#endif // COBWEB_HYBRID_NODE_H
