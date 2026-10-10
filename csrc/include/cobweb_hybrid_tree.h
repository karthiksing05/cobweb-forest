#ifndef COBWEB_HYBRID_TREE_H
#define COBWEB_HYBRID_TREE_H

#include <climits>
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <random>
#include <Eigen/Dense>

#include "cobweb_discrete_types.h"

class CobwebHybridNode;

// How expanded nodes are scored (queue order AND mixture weight) in the
// best-first multi-node schemes ``categorize`` / ``predict`` / ``log_prob``:
//   LIKELIHOOD  log P(x|c)                      what the C++ discrete / continuous trees do
//   PRIOR       log P(x|c) + log P(c)            global prior count_c / count_root
//   PATH        log P(c|x) + log P(x|c)          cobweb_torch: P(c|x) is the path posterior,
//                                                i.e. the child posteriors (likelihood x
//                                                count prior, normalised over the siblings)
//                                                multiplied down the path from the root
//   POSTERIOR   log P(c|x)                       the path posterior alone (the variant left
//                                                commented out in cobweb_torch)
enum class NodeWeighting : int { LIKELIHOOD = 0, PRIOR = 1, PATH = 2, POSTERIOR = 3 };

// A Cobweb tree over instances that are (discrete attribute-value table,
// continuous vector) pairs.  See cobweb_hybrid_node.h for the scoring.
class CobwebHybridTree {
    public:
        int size;            // dimensionality of the continuous block (0 = none)
        float alpha;         // Dirichlet smoothing for the discrete block
        float prior_var;     // variance floor / regulariser for the continuous block
        float cont_weight;   // multiplier on the continuous entropy block
        bool weight_attr;    // weight discrete attrs by root a_count / root count
        std::unordered_map<ATTR_TYPE, double> attr_weights;
        int depth_max;
        int branch_max;
        // Cobweb/4L's add-only training (Lian, Wang & MacLellan 2025): at each
        // branch only the "add to best child" operation is considered -- no
        // new / merge / split -- while fringe splits still create leaves.
        bool add_only;
        bool allow_new;      // with add_only: also consider creating a new child
        // Tie-breaking RNG.  Unseeded, the tree uses the module-global stream
        // (custom_rand); tree.set_random_seed(s) gives it a private stream so
        // its tie-breaks depend only on its own history (needed for
        // process-per-tree training to reproduce serial training exactly).
        bool own_rng;
        std::mt19937_64 rng;
        std::uniform_real_distribution<double> unif{0.0, 1.0};
        double rand();
        void set_random_seed(uint64_t seed);
        CobwebHybridNode *root;
        AV_KEY_TYPE attr_vals;

        CobwebHybridTree(int size, float alpha = 1.0f, float prior_var = 1.0f,
                         float cont_weight = 1.0f, bool weight_attr = true,
                         int depth_max = 1000, int branch_max = 1000,
                         std::unordered_map<ATTR_TYPE, double> attr_weights = {});
        ~CobwebHybridTree();

        void clear();

        CobwebHybridNode *ifit(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont);
        void fit(std::vector<INSTANCE_TYPE> discs, std::vector<Eigen::VectorXf> conts,
                 int iterations = 1, bool randomizeFirst = false);
        CobwebHybridNode *cobweb(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont);

        // Root-to-leaf descent by log P(c | x); no tree modification.  (This
        // is ``categorize`` in the C++ discrete / continuous trees and
        // ``old_categorize`` in cobweb_torch.)
        CobwebHybridNode *get_leaf(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont);
        // Root-to-leaf descent by relative partition utility (two_best_children).
        CobwebHybridNode *categorize_with_cu(const INSTANCE_TYPE &disc,
                const Eigen::VectorXf &cont);

        // --- multi-node schemes (cobweb_torch's _cobweb_categorize / _predict_probs) ---
        // Shared best-first expansion.  ``visit(score, path_ll, node)`` is
        // called for every expanded node (the root first); return false to
        // stop.  ``score`` is the node's weight under ``weighting`` and
        // ``path_ll`` its path posterior log P(c | x) (0 at the root).
        void expand(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                int max_nodes, bool greedy, NodeWeighting weighting,
                double eval_alpha, double eval_prior_var,
                const std::function<bool(double, double, CobwebHybridNode *)> &visit);

        // Best-first search for the single concept that best explains x.
        // use_best: return the highest-scoring expanded node (which may be an
        //           internal node) and stop early, branch-and-bound style, as
        //           soon as the best score exceeds the path posterior of the
        //           node being expanded (only sound for PATH weighting, where
        //           descendants cannot beat their ancestor's path posterior).
        //           With use_best = false the last expanded node is returned
        //           (with greedy = true: the leaf of the greedy descent).
        CobwebHybridNode *categorize(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                bool use_best = true, bool greedy = false, int max_nodes = INT_MAX,
                NodeWeighting weighting = NodeWeighting::PATH);

        // Tree-mixture estimates: every expanded node contributes with weight
        // proportional to exp(score).  ``predict`` returns (discrete value
        // probabilities per attribute, continuous mean); ``log_prob`` the
        // weighted average of the nodes' log P(x | c).
        double log_prob(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                int max_nodes, bool greedy = false, double eval_alpha = -1.0,
                double eval_prior_var = -1.0,
                NodeWeighting weighting = NodeWeighting::PATH);
        // leaves_only (forest iteration): only expanded leaves enter the
        // mixture (internal nodes still steer the best-first expansion).
        std::tuple<INSTANCE_TYPE, Eigen::VectorXf> predict(const INSTANCE_TYPE &disc,
                const Eigen::VectorXf &cont, int max_nodes, bool greedy = false,
                NodeWeighting weighting = NodeWeighting::PATH, bool leaves_only = false);
        // Continuous block only (no discrete table is built): the mixture mean.
        Eigen::VectorXf predict_mean(const INSTANCE_TYPE &disc,
                const Eigen::VectorXf &cont, int max_nodes, bool greedy = false,
                NodeWeighting weighting = NodeWeighting::PATH, bool leaves_only = false);
        // The expanded nodes with their scores: (score, path posterior, node).
        std::vector<std::tuple<double, double, CobwebHybridNode *>> expanded(
                const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont, int max_nodes,
                bool greedy = false, NodeWeighting weighting = NodeWeighting::PATH);
        // The same mixture for ONE discrete attribute, computed sparsely: the
        // cost per expanded node is the number of values the node stores
        // for ``attr`` rather than the whole value space of every attribute.
        std::unordered_map<VALUE_TYPE, double> predict_attr(const INSTANCE_TYPE &disc,
                const Eigen::VectorXf &cont, ATTR_TYPE attr, int max_nodes,
                bool greedy = false, NodeWeighting weighting = NodeWeighting::PATH);

        void set_attr_weight(ATTR_TYPE attr, double weight);
        int num_concepts();

        void dump_json(const std::string &filename);
        void load_json(const std::string &filename);
};

#endif // COBWEB_HYBRID_TREE_H
