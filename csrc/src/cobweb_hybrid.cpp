#include <nanobind/nanobind.h>

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

#include "cobweb_hybrid_node.h"
#include "cobweb_hybrid_tree.h"
#include "helper.h"

#include <climits>
#include <stdexcept>

namespace nb = nanobind;

// "likelihood" | "prior" | "path" -> NodeWeighting (see cobweb_hybrid_tree.h).
static NodeWeighting parse_weighting(const std::string &name)
{
    if (name == "likelihood") return NodeWeighting::LIKELIHOOD;
    if (name == "prior") return NodeWeighting::PRIOR;
    if (name == "path") return NodeWeighting::PATH;
    if (name == "posterior") return NodeWeighting::POSTERIOR;
    throw std::invalid_argument("weighting must be 'likelihood', 'prior', 'path' or 'posterior', got '" + name + "'");
}

NB_MODULE(_core, m)
{
    m.doc() = "cobweb hybrid: concepts with a discrete attribute-value table "
              "AND a continuous (diagonal Gaussian) block, scored with one "
              "information-theoretic partition utility";

    // Deterministically seed the C++ RNG used for tie-breaking in
    // two_best_children / get_best_operation (see helper.h).
    m.def("set_random_seed", &set_random_seed, nb::arg("seed"),
          "Seed cobweb's internal tie-breaking RNG for reproducibility.");

    nb::class_<CobwebHybridNode>(m, "CobwebHybridNode")
        .def(nb::init<int>(), nb::arg("size"))
        // --- entropies ---
        .def("entropy", &CobwebHybridNode::entropy)
        .def("entropy_discrete", &CobwebHybridNode::entropy_discrete)
        .def("entropy_continuous", &CobwebHybridNode::entropy_continuous)
        .def("entropy_attr", &CobwebHybridNode::entropy_attr, nb::arg("attr"))
        .def("partition_utility", &CobwebHybridNode::partition_utility)
        // ifit early-stop signals: PU of inserting into the best child vs.
        // PU of creating a new child.  ``new > best`` means the instance
        // does not fit any existing cluster well (a novelty signal).
        .def("pu_for_best_insert",
                [](CobwebHybridNode &self, const INSTANCE_TYPE &disc,
                   const Eigen::VectorXf &cont) -> double {
                    if (self.children.empty()) return 0.0;
                    return std::get<0>(self.two_best_children(disc, cont));
                },
                nb::arg("disc"), nb::arg("cont"))
        .def("pu_for_new_child",
                [](CobwebHybridNode &self, const INSTANCE_TYPE &disc,
                   const Eigen::VectorXf &cont) -> double {
                    return self.pu_for_new_child(disc, cont);
                },
                nb::arg("disc"), nb::arg("cont"))
        // --- probabilities ---
        .def("log_prob_instance", &CobwebHybridNode::log_prob_instance,
                nb::arg("disc"), nb::arg("cont"),
                nb::arg("eval_alpha") = -1.0, nb::arg("eval_prior_var") = -1.0)
        .def("log_prob_class_given_instance", &CobwebHybridNode::log_prob_class_given_instance,
                nb::arg("disc"), nb::arg("cont"))
        .def("log_prob_children_given_instance", &CobwebHybridNode::log_prob_children_given_instance,
                nb::arg("disc"), nb::arg("cont"))
        .def("predict_probs", &CobwebHybridNode::predict_probs)
        .def("prob", &CobwebHybridNode::prob, nb::arg("attr"), nb::arg("val"))
        // Model variance of the continuous block: sum_sq / count + prior_var.
        .def("variance",
                [](CobwebHybridNode &self, double eval_prior_var) -> Eigen::VectorXf {
                    return self.model_variance(eval_prior_var).matrix();
                },
                nb::arg("eval_prior_var") = -1.0)
        // --- basic level ---
        .def("expected_pmi", &CobwebHybridNode::expected_pmi,
                nb::arg("eval_alpha") = -1.0, nb::arg("eval_prior_var") = -1.0)
        .def("expected_pmi_discrete", &CobwebHybridNode::expected_pmi_discrete,
                nb::arg("eval_alpha") = -1.0)
        .def("expected_pmi_continuous", &CobwebHybridNode::expected_pmi_continuous,
                nb::arg("eval_prior_var") = -1.0)
        .def("get_basic", &CobwebHybridNode::get_basic, nb::rv_policy::reference,
                nb::arg("eval_alpha") = -1.0, nb::arg("eval_prior_var") = -1.0)
        // --- misc ---
        .def("concept_hash", &CobwebHybridNode::concept_hash)
        .def("depth", &CobwebHybridNode::depth)
        .def("num_concepts", &CobwebHybridNode::num_concepts)
        .def("is_parent", &CobwebHybridNode::is_parent, nb::arg("other"))
        .def("__hash__", &CobwebHybridNode::_hash)
        .def_ro("uid", &CobwebHybridNode::uid)
        .def_ro("count", &CobwebHybridNode::count)
        .def_ro("children", &CobwebHybridNode::children, nb::rv_policy::reference)
        .def_ro("parent", &CobwebHybridNode::parent, nb::rv_policy::reference)
        .def_ro("tree", &CobwebHybridNode::tree, nb::rv_policy::reference)
        .def_ro("av_count", &CobwebHybridNode::av_count)
        .def_ro("a_count", &CobwebHybridNode::a_count)
        .def_ro("mean", &CobwebHybridNode::mean, nb::rv_policy::copy)
        .def_ro("sum_sq", &CobwebHybridNode::sum_sq, nb::rv_policy::copy);

    nb::class_<CobwebHybridTree>(m, "CobwebHybridTree")
        .def(nb::init<int, float, float, float, bool, int, int,
                      std::unordered_map<ATTR_TYPE, double>>(),
                nb::arg("size"),
                nb::arg("alpha") = 1.0f,
                nb::arg("prior_var") = 1.0f,
                nb::arg("cont_weight") = 1.0f,
                nb::arg("weight_attr") = true,
                nb::arg("depth_max") = 1000,
                nb::arg("branch_max") = 1000,
                nb::arg("attr_weights") = std::unordered_map<ATTR_TYPE, double>{},
                "size: dimensionality of the continuous block (0 = discrete only).\n"
                "alpha: Dirichlet smoothing of the discrete block.\n"
                "prior_var: variance floor of the continuous block (its 'alpha').\n"
                "cont_weight: multiplier on the continuous entropy in the partition utility.")
        .def("ifit", &CobwebHybridTree::ifit,
                nb::arg("disc"), nb::arg("cont"),
                nb::rv_policy::reference_internal)
        .def("fit", &CobwebHybridTree::fit,
                nb::arg("discs"), nb::arg("conts"),
                nb::arg("iterations") = 1, nb::arg("randomizeFirst") = false)
        .def("get_leaf", &CobwebHybridTree::get_leaf,
                nb::arg("disc"), nb::arg("cont"),
                nb::rv_policy::reference,
                "Root-to-leaf descent by log P(c|x) (the C++ trees' categorize).")
        .def("categorize_with_cu", &CobwebHybridTree::categorize_with_cu,
                nb::arg("disc"), nb::arg("cont"),
                nb::rv_policy::reference)
        // --- multi-node schemes (cobweb_torch style) ---
        // weighting: "path" (log P(c|x) + log P(x|c), cobweb_torch), "posterior"
        // (log P(c|x) alone), "prior" (log P(x|c) + log P(c)) or "likelihood"
        // (log P(x|c), the C++ trees).
        .def("categorize",
                [](CobwebHybridTree &self, const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                   bool use_best, bool greedy, int max_nodes, const std::string &weighting) {
                    return self.categorize(disc, cont, use_best, greedy, max_nodes,
                                           parse_weighting(weighting));
                },
                nb::arg("disc"), nb::arg("cont"), nb::arg("use_best") = true,
                nb::arg("greedy") = false, nb::arg("max_nodes") = INT_MAX,
                nb::arg("weighting") = "path",
                nb::rv_policy::reference,
                "Best-first search for the concept that best explains the instance "
                "(cobweb_torch's categorize).  use_best returns the highest-scoring "
                "expanded node (possibly internal) with branch-and-bound early stop; "
                "otherwise the last expanded node (with greedy: the leaf reached).")
        .def("log_prob",
                [](CobwebHybridTree &self, const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                   int max_nodes, bool greedy, double eval_alpha, double eval_prior_var,
                   const std::string &weighting) {
                    return self.log_prob(disc, cont, max_nodes, greedy, eval_alpha, eval_prior_var,
                                         parse_weighting(weighting));
                },
                nb::arg("disc"), nb::arg("cont"), nb::arg("max_nodes"),
                nb::arg("greedy") = false,
                nb::arg("eval_alpha") = -1.0, nb::arg("eval_prior_var") = -1.0,
                nb::arg("weighting") = "path")
        .def("predict",
                [](CobwebHybridTree &self, const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                   int max_nodes, bool greedy, const std::string &weighting, bool leaves_only) {
                    return self.predict(disc, cont, max_nodes, greedy, parse_weighting(weighting),
                                        leaves_only);
                },
                nb::arg("disc"), nb::arg("cont"), nb::arg("max_nodes"),
                nb::arg("greedy") = false, nb::arg("weighting") = "path",
                nb::arg("leaves_only") = false,
                "Multi-node tree-mixture prediction (cobweb_torch's predict_probs).  "
                "Returns (probs, mean): the discrete value probabilities per attribute "
                "and the continuous mean, mixing the expanded nodes with weights "
                "proportional to exp(score).  Pass an empty cont array to treat the "
                "continuous value as missing.")
        .def("predict_attr",
                [](CobwebHybridTree &self, const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                   ATTR_TYPE attr, int max_nodes, bool greedy, const std::string &weighting) {
                    return self.predict_attr(disc, cont, attr, max_nodes, greedy, parse_weighting(weighting));
                },
                nb::arg("disc"), nb::arg("cont"), nb::arg("attr"), nb::arg("max_nodes"),
                nb::arg("greedy") = false, nb::arg("weighting") = "path",
                "predict() for one discrete attribute, computed sparsely (same numbers, "
                "cost per node = values the node stores for attr instead of the whole value space).")
        .def("predict_mean",
                [](CobwebHybridTree &self, const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                   int max_nodes, bool greedy, const std::string &weighting, bool leaves_only) {
                    return self.predict_mean(disc, cont, max_nodes, greedy,
                                             parse_weighting(weighting), leaves_only);
                },
                nb::arg("disc"), nb::arg("cont"), nb::arg("max_nodes"),
                nb::arg("greedy") = false, nb::arg("weighting") = "path",
                nb::arg("leaves_only") = false,
                "Mixture mean of the continuous block; NaN entries of cont are missing.")
        .def("expanded",
                [](CobwebHybridTree &self, const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
                   int max_nodes, bool greedy, const std::string &weighting) {
                    return self.expanded(disc, cont, max_nodes, greedy, parse_weighting(weighting));
                },
                nb::arg("disc"), nb::arg("cont"), nb::arg("max_nodes"),
                nb::arg("greedy") = false, nb::arg("weighting") = "path",
                nb::rv_policy::reference_internal,
                "Expanded nodes as (score, path posterior log P(c|x), node), in expansion order.")
        .def("set_attr_weight", &CobwebHybridTree::set_attr_weight,
                nb::arg("attr"), nb::arg("weight"))
        .def("num_concepts", &CobwebHybridTree::num_concepts)
        .def("clear", &CobwebHybridTree::clear)
        .def("dump_json", &CobwebHybridTree::dump_json, nb::arg("filename"))
        .def("load_json", &CobwebHybridTree::load_json, nb::arg("filename"))
        .def_ro("root", &CobwebHybridTree::root, nb::rv_policy::reference)
        .def_ro("size", &CobwebHybridTree::size)
        .def_ro("alpha", &CobwebHybridTree::alpha)
        .def_rw("prior_var", &CobwebHybridTree::prior_var)
        .def_rw("cont_weight", &CobwebHybridTree::cont_weight)
        .def_ro("weight_attr", &CobwebHybridTree::weight_attr)
        .def_ro("depth_max", &CobwebHybridTree::depth_max)
        .def_ro("branch_max", &CobwebHybridTree::branch_max)
        .def_rw("add_only", &CobwebHybridTree::add_only,
                "Cobweb/4L add-only training: only the add operation is considered at "
                "each branch (no new / merge / split); fringe splits still make leaves.")
        .def("set_random_seed", &CobwebHybridTree::set_random_seed, nb::arg("seed"),
             "Give this tree a private tie-breaking RNG seeded with `seed` (otherwise the module-global one is used)")
        .def_rw("allow_new", &CobwebHybridTree::allow_new,
                "With add_only: also consider the NEW (create child) operation at internal "
                "nodes. Nothing is deleted or re-parented, so stored paths stay valid.")
        .def_ro("attr_vals", &CobwebHybridTree::attr_vals)
        .def_rw("attr_weights", &CobwebHybridTree::attr_weights);
}
