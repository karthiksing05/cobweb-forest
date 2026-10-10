#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stack>
#include <stdexcept>
#include <unordered_map>

#include <rapidjson/writer.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/reader.h>
#include <rapidjson/istreamwrapper.h>

#include "helper.h"
#include "cobweb_hybrid_node.h"
#include "cobweb_hybrid_tree.h"

// With leaves_only the budget ``max_nodes`` counts LEAVES: expansion continues until that
// many leaves have entered the mixture (the M best leaf concepts), with at most
// LEAF_BUDGET_FACTOR * max_nodes expansions in total.  Counting all nodes instead can leave
// the mixture with no leaf at all in deep trees, and the prediction degenerates to zero.
static const int LEAF_BUDGET_FACTOR = 64;
static int leaf_budget_cap(int max_nodes)
{
    return (max_nodes >= INT_MAX / LEAF_BUDGET_FACTOR) ? INT_MAX : max_nodes * LEAF_BUDGET_FACTOR;
}


CobwebHybridTree::CobwebHybridTree(int size, float alpha, float prior_var, float cont_weight,
                                   bool weight_attr, int depth_max, int branch_max,
                                   std::unordered_map<ATTR_TYPE, double> attr_weights)
    : size(size),
    alpha(alpha),
    prior_var(prior_var),
    cont_weight(cont_weight),
    weight_attr(weight_attr),
    attr_weights(std::move(attr_weights)),
    depth_max(depth_max),
    branch_max(branch_max),
    add_only(false),
    allow_new(false),
    own_rng(false),
    root(nullptr)
{
    if (size < 0) throw std::invalid_argument("size must be >= 0");
    this->clear();
}

CobwebHybridTree::~CobwebHybridTree()
{
    delete this->root;
}

void CobwebHybridTree::clear()
{
    delete this->root;
    this->root = new CobwebHybridNode(this->size);
    this->root->tree = this;
    this->attr_vals = AV_KEY_TYPE();
}

void CobwebHybridTree::set_attr_weight(ATTR_TYPE attr, double weight)
{
    if (weight == 1.0) {
        this->attr_weights.erase(attr);
    } else {
        this->attr_weights[attr] = weight;
    }
}

int CobwebHybridTree::num_concepts()
{
    return this->root->num_concepts();
}

CobwebHybridNode *CobwebHybridTree::ifit(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont)
{
    return this->cobweb(disc, cont);
}

void CobwebHybridTree::fit(std::vector<INSTANCE_TYPE> discs, std::vector<Eigen::VectorXf> conts,
                           int iterations, bool randomizeFirst)
{
    if (discs.size() != conts.size()) {
        throw std::invalid_argument("fit: discs and conts must have the same length");
    }
    std::vector<size_t> order(discs.size());
    std::iota(order.begin(), order.end(), 0);
    std::default_random_engine engine;

    for (int i = 0; i < iterations; i++) {
        if (i > 0 || randomizeFirst) {
            std::shuffle(order.begin(), order.end(), engine);
        }
        for (auto idx : order) {
            this->ifit(discs[idx], conts[idx]);
        }
    }
}

CobwebHybridNode *CobwebHybridTree::cobweb(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont)
{
    if (this->size > 0 && cont.size() != this->size) {
        throw std::invalid_argument(
            "ifit: continuous vector has size " + std::to_string(cont.size()) +
            " but the tree expects " + std::to_string(this->size));
    }
    if (this->size > 0 && cont.hasNaN()) {
        throw std::invalid_argument("ifit: NaN (missing) continuous values are only allowed in queries");
    }

    for (auto &[attr, val_map] : disc) {
        for (auto &[val, cnt] : val_map) {
            this->attr_vals[attr].insert(val);
        }
    }

    CobwebHybridNode *current = this->root;
    int depth = 1;
    bool placed = false;

    while (depth < this->depth_max) {
        if (current->children.empty() &&
                (current->count == 0 || current->is_exact_match(disc, cont))) {
            // empty / exact match
            current->increment_counts(disc, cont);
            placed = true;
            break;
        }
        else if (current->children.empty()) {
            // fringe split: the leaf becomes an internal node with the old
            // leaf and the new instance as its two children.
            CobwebHybridNode *new_node = new CobwebHybridNode(current);
            current->parent = new_node;
            new_node->children.push_back(current);

            if (new_node->parent == nullptr) {
                this->root = new_node;
            } else {
                new_node->parent->children.erase(
                    std::remove(new_node->parent->children.begin(),
                                new_node->parent->children.end(), current),
                    new_node->parent->children.end());
                new_node->parent->children.push_back(new_node);
            }
            new_node->increment_counts(disc, cont);

            CobwebHybridNode *new_child = new CobwebHybridNode(this->size);
            new_child->parent = new_node;
            new_child->tree = this;
            new_child->increment_counts(disc, cont);
            new_node->children.push_back(new_child);

            current = new_child;
            placed = true;
            break;
        }
        else {
            auto [best1_pu, best1, best2] = current->two_best_children(disc, cont);
            auto [_, bestAction] = current->get_best_operation(disc, cont, best1, best2, best1_pu);

            if (bestAction == Operation::BEST) {
                current->increment_counts(disc, cont);
                current = best1;
            }
            else if (bestAction == Operation::NEW) {
                current->increment_counts(disc, cont);
                CobwebHybridNode *new_child = new CobwebHybridNode(this->size);
                new_child->parent = current;
                new_child->tree = this;
                new_child->increment_counts(disc, cont);
                current->children.push_back(new_child);
                current = new_child;
                placed = true;
                break;
            }
            else if (bestAction == Operation::MERGE) {
                current->increment_counts(disc, cont);
                CobwebHybridNode *new_child = new CobwebHybridNode(this->size);
                new_child->parent = current;
                new_child->tree = this;

                new_child->update_counts_from_node(best1);
                new_child->update_counts_from_node(best2);
                best1->parent = new_child;
                best2->parent = new_child;
                new_child->children.push_back(best1);
                new_child->children.push_back(best2);
                current->children.erase(
                    std::remove(current->children.begin(), current->children.end(), best1),
                    current->children.end());
                current->children.erase(
                    std::remove(current->children.begin(), current->children.end(), best2),
                    current->children.end());
                current->children.push_back(new_child);
                current = new_child;
            }
            else if (bestAction == Operation::SPLIT) {
                current->children.erase(
                    std::remove(current->children.begin(), current->children.end(), best1),
                    current->children.end());
                for (auto *c : best1->children) {
                    c->parent = current;
                    c->tree = this;
                    current->children.push_back(c);
                }
                best1->children.clear();
                delete best1;
            }
            else {
                throw std::runtime_error("unrecognised cobweb operation");
            }
        }
        depth++;
    }

    if (!placed) {
        // depth_max was reached without a normal exit.  Force-insert as a
        // fresh leaf so the returned pointer is never an internal node
        // (internal nodes can be deleted by a future SPLIT).
        current->increment_counts(disc, cont);
        if (!current->children.empty()) {
            CobwebHybridNode *new_leaf = new CobwebHybridNode(this->size);
            new_leaf->parent = current;
            new_leaf->tree = this;
            new_leaf->increment_counts(disc, cont);
            current->children.push_back(new_leaf);
            current = new_leaf;
        }
    }

    return current;
}

CobwebHybridNode *CobwebHybridTree::get_leaf(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont)
{
    CobwebHybridNode *current = this->root;

    while (!current->children.empty()) {
        CobwebHybridNode *best = nullptr;
        double best_logp = 0.0;
        for (auto *child : current->children) {
            double logp = child->log_prob_class_given_instance(disc, cont);
            if (best == nullptr || logp > best_logp) {
                best = child;
                best_logp = logp;
            }
        }
        current = best;
    }
    return current;
}

CobwebHybridNode *CobwebHybridTree::categorize_with_cu(const INSTANCE_TYPE &disc,
        const Eigen::VectorXf &cont)
{
    // Make sure the instance's values are part of the value space that the
    // entropy computations iterate over (as ifit would do).
    for (auto &[attr, val_map] : disc) {
        for (auto &[val, cnt] : val_map) {
            this->attr_vals[attr].insert(val);
        }
    }

    CobwebHybridNode *current = this->root;
    while (!current->children.empty()) {
        auto [best1_pu, best1, best2] = current->two_best_children(disc, cont);
        if (best1 == nullptr) break;
        current = best1;
    }
    return current;
}

void CobwebHybridTree::expand(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
        int max_nodes, bool greedy, NodeWeighting weighting,
        double eval_alpha, double eval_prior_var,
        const std::function<bool(double, double, CobwebHybridNode *)> &visit)
{
    // Queue entries: (score, path posterior log P(c|x), random tie-break, node).
    auto queue = std::priority_queue<std::tuple<double, double, double, CobwebHybridNode *>>();
    double root_ll = this->root->log_prob_instance(disc, cont, eval_alpha, eval_prior_var);
    queue.push(std::make_tuple(root_ll, 0.0, this->rand(), this->root));
    int nodes_visited = 0;

    while (!queue.empty()) {
        auto [score, path_ll, tie, curr] = queue.top();
        queue.pop();
        nodes_visited += 1;

        if (!visit(score, path_ll, curr)) break;

        if (greedy) {
            queue = std::priority_queue<std::tuple<double, double, double, CobwebHybridNode *>>();
        }
        if (nodes_visited >= max_nodes) break;
        if (curr->children.empty()) continue;

        // Child posteriors given the instance, normalised over the siblings:
        // P(child | x, curr) ∝ P(x | child) * count_child / count_curr.
        size_t n = curr->children.size();
        std::vector<double> ll_inst(n), unnorm(n);
        for (size_t i = 0; i < n; ++i) {
            CobwebHybridNode *child = curr->children[i];
            ll_inst[i] = child->log_prob_instance(disc, cont, eval_alpha, eval_prior_var);
            unnorm[i] = ll_inst[i] + std::log(child->count / curr->count);
        }
        double log_p_of_x = logsumexp(unnorm);

        for (size_t i = 0; i < n; ++i) {
            CobwebHybridNode *child = curr->children[i];
            double child_path_ll = unnorm[i] - log_p_of_x + path_ll;
            double child_score;
            switch (weighting) {
                case NodeWeighting::LIKELIHOOD:
                    child_score = ll_inst[i];
                    break;
                case NodeWeighting::PRIOR:
                    child_score = ll_inst[i] + std::log(child->count / this->root->count);
                    break;
                case NodeWeighting::POSTERIOR:
                    child_score = child_path_ll;                // log P(c|x)
                    break;
                case NodeWeighting::PATH:
                default:
                    child_score = child_path_ll + ll_inst[i];   // log P(c|x) + log P(x|c)
                    break;
            }
            queue.push(std::make_tuple(child_score, child_path_ll, this->rand(), child));
        }
    }
}

CobwebHybridNode *CobwebHybridTree::categorize(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
        bool use_best, bool greedy, int max_nodes, NodeWeighting weighting)
{
    CobwebHybridNode *best = this->root;
    CobwebHybridNode *last = this->root;
    double best_score = -std::numeric_limits<double>::infinity();

    this->expand(disc, cont, max_nodes, greedy, weighting, -1.0, -1.0,
        [&](double score, double path_ll, CobwebHybridNode *node) {
            last = node;
            if (score > best_score) {
                best = node;
                best_score = score;
            }
            // Branch and bound (cobweb_torch): under PATH weighting every
            // descendant's score is at most its ancestor's path posterior
            // (plus a log-likelihood that is <= 0 for the discrete block), so
            // once the best score exceeds this node's path posterior nothing
            // below it can win.
            if (use_best && (weighting == NodeWeighting::PATH || weighting == NodeWeighting::POSTERIOR) &&
                    best_score > path_ll) return false;
            return true;
        });

    return use_best ? best : last;
}

double CobwebHybridTree::log_prob(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
        int max_nodes, bool greedy, double eval_alpha, double eval_prior_var,
        NodeWeighting weighting)
{
    // Weighted average of the expanded nodes' log P(x | c) with weights
    // proportional to exp(score) (mirrors CobwebDiscreteTree::log_prob).
    double total_weight = 0.0;
    bool first = true;
    double out = 0.0;

    this->expand(disc, cont, max_nodes, greedy, weighting, eval_alpha, eval_prior_var,
        [&](double score, double path_ll, CobwebHybridNode *node) {
            (void)path_ll;
            if (first) {
                total_weight = score;
                first = false;
            } else {
                total_weight = logsumexp(total_weight, score);
            }
            double ll = node->log_prob_instance(disc, cont, eval_alpha, eval_prior_var);
            out += std::exp(score - total_weight) * (ll - out);
            return true;
        });

    return out;
}

std::tuple<INSTANCE_TYPE, Eigen::VectorXf> CobwebHybridTree::predict(const INSTANCE_TYPE &disc,
        const Eigen::VectorXf &cont, int max_nodes, bool greedy, NodeWeighting weighting,
        bool leaves_only)
{
    // Tree-mixture prediction (cobweb_torch's _predict_probs): each expanded
    // node contributes its value probabilities and its continuous mean with
    // weight exp(score) / sum of exp(score) over the expanded nodes.  The
    // running form ``out += w * (p - out)`` with w = exp(score - logsumexp so
    // far) is that normalised mixture computed online.
    INSTANCE_TYPE out;
    for (auto &[attr, val_set] : this->attr_vals) {
        if (attr < 0) continue;
        for (auto val : val_set) out[attr][val] = 0.0;
    }
    Eigen::VectorXf mean_out = Eigen::VectorXf::Zero(this->size);

    double total_weight = 0.0;
    bool first = true;
    int leaves = 0;

    this->expand(disc, cont, leaves_only ? leaf_budget_cap(max_nodes) : max_nodes, greedy,
        weighting, -1.0, -1.0,
        [&](double score, double path_ll, CobwebHybridNode *node) {
            (void)path_ll;
            if (leaves_only && !node->children.empty()) return true;
            if (first) {
                total_weight = score;
                first = false;
            } else {
                total_weight = logsumexp(total_weight, score);
            }
            double w = std::exp(score - total_weight);

            INSTANCE_TYPE probs = node->predict_probs();
            for (auto &[attr, val_map] : probs) {
                for (auto &[val, p] : val_map) {
                    out[attr][val] += w * (p - out[attr][val]);
                }
            }
            if (this->size > 0) {
                mean_out += static_cast<float>(w) * (node->mean - mean_out);
            }
            return !(leaves_only && ++leaves >= max_nodes);
        });

    return std::make_tuple(out, mean_out);
}

Eigen::VectorXf CobwebHybridTree::predict_mean(const INSTANCE_TYPE &disc,
        const Eigen::VectorXf &cont, int max_nodes, bool greedy, NodeWeighting weighting,
        bool leaves_only)
{
    Eigen::VectorXf mean_out = Eigen::VectorXf::Zero(this->size);
    double total_weight = 0.0;
    bool first = true;
    int leaves = 0;
    this->expand(disc, cont, leaves_only ? leaf_budget_cap(max_nodes) : max_nodes, greedy,
        weighting, -1.0, -1.0,
        [&](double score, double path_ll, CobwebHybridNode *node) {
            (void)path_ll;
            if (leaves_only && !node->children.empty()) return true;
            if (first) {
                total_weight = score;
                first = false;
            } else {
                total_weight = logsumexp(total_weight, score);
            }
            double w = std::exp(score - total_weight);
            mean_out += static_cast<float>(w) * (node->mean - mean_out);
            return !(leaves_only && ++leaves >= max_nodes);
        });
    if (first) mean_out = this->root->mean;   // no node entered the mixture: the root's mean
    return mean_out;
}

std::vector<std::tuple<double, double, CobwebHybridNode *>> CobwebHybridTree::expanded(
        const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont, int max_nodes,
        bool greedy, NodeWeighting weighting)
{
    std::vector<std::tuple<double, double, CobwebHybridNode *>> out;
    this->expand(disc, cont, max_nodes, greedy, weighting, -1.0, -1.0,
        [&](double score, double path_ll, CobwebHybridNode *node) {
            out.emplace_back(score, path_ll, node);
            return true;
        });
    return out;
}

std::unordered_map<VALUE_TYPE, double> CobwebHybridTree::predict_attr(const INSTANCE_TYPE &disc,
        const Eigen::VectorXf &cont, ATTR_TYPE attr, int max_nodes, bool greedy,
        NodeWeighting weighting)
{
    // Sparse version of ``predict`` for a single attribute.  With
    // P(v | c) = (av_c(v) + alpha) / (a_count_c + |V| alpha) the mixture
    //   sum_c w_c P(v | c) = sum_c w_c alpha / denom_c  +  sum_c w_c av_c(v) / denom_c
    // has a value-independent first term and a second term that only touches
    // the values a node actually stores.  The nodes are collected first (no
    // tree modification happens during expansion) so the weights can be
    // normalised with one logsumexp.
    std::unordered_map<VALUE_TYPE, double> out;
    auto kit = this->attr_vals.find(attr);
    if (kit == this->attr_vals.end()) return out;
    const double alpha = this->alpha;
    const double num_vals = static_cast<double>(kit->second.size());

    std::vector<std::pair<double, CobwebHybridNode *>> expanded;
    this->expand(disc, cont, max_nodes, greedy, weighting, -1.0, -1.0,
        [&](double score, double path_ll, CobwebHybridNode *node) {
            (void)path_ll;
            expanded.emplace_back(score, node);
            return true;
        });

    std::vector<double> scores;
    scores.reserve(expanded.size());
    for (auto &[score, node] : expanded) scores.push_back(score);
    double log_norm = logsumexp(scores);

    double base = 0.0;
    for (auto &[score, node] : expanded) {
        double w = std::exp(score - log_norm);
        double attr_count = 0.0;
        auto acit = node->a_count.find(attr);
        if (acit != node->a_count.end()) attr_count = acit->second;
        double denom = attr_count + num_vals * alpha;
        base += w * alpha / denom;
        auto ait = node->av_count.find(attr);
        if (ait == node->av_count.end()) continue;
        for (auto &[val, av] : ait->second) {
            out[val] += w * av / denom;
        }
    }
    for (auto val : kit->second) out[val] += base;
    return out;
}

// ============================================================
// JSON serialisation
// ============================================================

void CobwebHybridTree::dump_json(const std::string &filename)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    rapidjson::OStreamWrapper osw(ofs);
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);

    writer.StartObject();

    writer.Key("tree_parameters");
    writer.StartObject();
    writer.Key("size");        writer.Int(this->size);
    writer.Key("alpha");       writer.Double(this->alpha);
    writer.Key("prior_var");   writer.Double(this->prior_var);
    writer.Key("cont_weight"); writer.Double(this->cont_weight);
    writer.Key("weight_attr"); writer.Bool(this->weight_attr);
    writer.Key("depth_max");   writer.Int(this->depth_max);
    writer.Key("branch_max");  writer.Int(this->branch_max);
    writer.EndObject();

    writer.Key("attr_weights");
    writer.StartObject();
    for (const auto &[attr, w] : this->attr_weights) {
        writer.Key(std::to_string(attr).c_str());
        writer.Double(w);
    }
    writer.EndObject();

    writer.Key("attr_vals");
    writer.StartObject();
    for (const auto &[attr, val_set] : this->attr_vals) {
        writer.Key(std::to_string(attr).c_str());
        writer.StartArray();
        for (const auto &val : val_set) writer.Int(val);
        writer.EndArray();
    }
    writer.EndObject();

    writer.Key("nodes");
    writer.StartArray();

    std::stack<std::tuple<CobwebHybridNode *, int>> stack;
    stack.emplace(this->root, -1);
    int node_id = 0;

    while (!stack.empty()) {
        auto [node, parent_id] = stack.top();
        stack.pop();

        writer.StartObject();
        writer.Key("id");        writer.Int(node_id);
        writer.Key("parent_id");
        if (parent_id == -1) writer.Null(); else writer.Int(parent_id);
        writer.Key("concept_id"); writer.String(node->concept_id.c_str());
        writer.Key("count");      writer.Double(node->count);

        writer.Key("a_count");
        writer.StartObject();
        for (const auto &[attr, c] : node->a_count) {
            writer.Key(std::to_string(attr).c_str());
            writer.Double(c);
        }
        writer.EndObject();

        writer.Key("sum_n_logn");
        writer.StartObject();
        for (const auto &[attr, s] : node->sum_n_logn) {
            writer.Key(std::to_string(attr).c_str());
            writer.Double(s);
        }
        writer.EndObject();

        writer.Key("av_count");
        writer.StartObject();
        for (const auto &[attr, val_map] : node->av_count) {
            writer.Key(std::to_string(attr).c_str());
            writer.StartObject();
            for (const auto &[val, c] : val_map) {
                writer.Key(std::to_string(val).c_str());
                writer.Double(c);
            }
            writer.EndObject();
        }
        writer.EndObject();

        writer.Key("mean");
        writer.StartArray();
        for (int i = 0; i < node->mean.size(); ++i) writer.Double(node->mean[i]);
        writer.EndArray();

        writer.Key("sum_sq");
        writer.StartArray();
        for (int i = 0; i < node->sum_sq.size(); ++i) writer.Double(node->sum_sq[i]);
        writer.EndArray();

        writer.EndObject();

        int current_id = node_id++;
        for (int i = static_cast<int>(node->children.size()) - 1; i >= 0; --i) {
            stack.emplace(node->children[i], current_id);
        }
    }

    writer.EndArray();
    writer.EndObject();
}

namespace {

    class HybridTreeSaxHandler final
        : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, HybridTreeSaxHandler> {
        public:
            CobwebHybridTree *tree;
            std::unordered_map<int, CobwebHybridNode *> id_to_node;
            std::unordered_map<int, int> child_to_parent;

            enum class Mode {
                ROOT,
                TREE_PARAMETERS,
                ATTR_WEIGHTS_OBJECT,
                ATTR_VALS_OBJECT,
                ATTR_VALS_ARRAY,
                NODES_ARRAY,
                NODE_OBJECT,
                NODE_ARRAY,          // "mean" / "sum_sq" inside a node
                A_COUNT_OBJECT,
                SUM_N_LOGN_OBJECT,
                AV_COUNT_OUTER_OBJECT,
                AV_COUNT_INNER_OBJECT
            };

            Mode mode = Mode::ROOT;
            std::string current_key;
            std::string array_key;       // which node array we're filling
            int current_id = -1;
            int current_parent_id = -2;
            int current_attr_key = -1;
            int current_map_key = -1;
            double node_count = 0.0;
            std::vector<float> temp_array;
            CobwebHybridNode *building_node = nullptr;

            explicit HybridTreeSaxHandler(CobwebHybridTree *t) : tree(t) {
                delete tree->root;
                tree->root = nullptr;
                tree->attr_vals.clear();
                tree->attr_weights.clear();
            }

            static int key_to_int(const std::string &s) { return std::stoi(s); }

            bool Key(const char *str, rapidjson::SizeType, bool) {
                current_key.assign(str);
                if (mode == Mode::A_COUNT_OBJECT || mode == Mode::SUM_N_LOGN_OBJECT ||
                        mode == Mode::AV_COUNT_OUTER_OBJECT || mode == Mode::AV_COUNT_INNER_OBJECT ||
                        mode == Mode::ATTR_VALS_OBJECT || mode == Mode::ATTR_WEIGHTS_OBJECT) {
                    current_map_key = key_to_int(current_key);
                    if (mode == Mode::AV_COUNT_OUTER_OBJECT) current_attr_key = current_map_key;
                }
                return true;
            }

            bool String(const char *str, rapidjson::SizeType length, bool) {
                if (mode == Mode::NODE_OBJECT && current_key == "concept_id" && building_node) {
                    building_node->concept_id = std::string(str, length);
                }
                return true;
            }

            bool StartObject() {
                if (mode == Mode::ROOT && current_key == "tree_parameters") { mode = Mode::TREE_PARAMETERS; return true; }
                if (mode == Mode::ROOT && current_key == "attr_weights")    { mode = Mode::ATTR_WEIGHTS_OBJECT; return true; }
                if (mode == Mode::ROOT && current_key == "attr_vals")       { mode = Mode::ATTR_VALS_OBJECT; return true; }
                if (mode == Mode::NODES_ARRAY) {
                    mode = Mode::NODE_OBJECT;
                    current_id = -1;
                    current_parent_id = -2;
                    node_count = 0.0;
                    building_node = new CobwebHybridNode(tree->size);
                    building_node->tree = tree;
                    return true;
                }
                if (mode == Mode::NODE_OBJECT && current_key == "a_count")    { mode = Mode::A_COUNT_OBJECT; return true; }
                if (mode == Mode::NODE_OBJECT && current_key == "sum_n_logn") { mode = Mode::SUM_N_LOGN_OBJECT; return true; }
                if (mode == Mode::NODE_OBJECT && current_key == "av_count")   { mode = Mode::AV_COUNT_OUTER_OBJECT; return true; }
                if (mode == Mode::AV_COUNT_OUTER_OBJECT)                      { mode = Mode::AV_COUNT_INNER_OBJECT; return true; }
                return true;
            }

            bool EndObject(rapidjson::SizeType) {
                switch (mode) {
                    case Mode::A_COUNT_OBJECT:
                    case Mode::SUM_N_LOGN_OBJECT:
                        mode = Mode::NODE_OBJECT; return true;
                    case Mode::AV_COUNT_INNER_OBJECT:
                        mode = Mode::AV_COUNT_OUTER_OBJECT; return true;
                    case Mode::AV_COUNT_OUTER_OBJECT:
                        mode = Mode::NODE_OBJECT; return true;
                    case Mode::ATTR_VALS_OBJECT:
                    case Mode::ATTR_WEIGHTS_OBJECT:
                    case Mode::TREE_PARAMETERS:
                        mode = Mode::ROOT; return true;
                    case Mode::NODE_OBJECT: {
                        if (!building_node) return false;
                        building_node->count = node_count;
                        id_to_node[current_id] = building_node;
                        if (current_parent_id == -1) tree->root = building_node;
                        else child_to_parent[current_id] = current_parent_id;
                        building_node = nullptr;
                        mode = Mode::NODES_ARRAY;
                        return true;
                    }
                    default:
                        return true;
                }
            }

            bool StartArray() {
                if (mode == Mode::ROOT && current_key == "nodes") {
                    mode = Mode::NODES_ARRAY;
                } else if (mode == Mode::ATTR_VALS_OBJECT) {
                    mode = Mode::ATTR_VALS_ARRAY;
                } else if (mode == Mode::NODE_OBJECT && (current_key == "mean" || current_key == "sum_sq")) {
                    mode = Mode::NODE_ARRAY;
                    array_key = current_key;
                    temp_array.clear();
                }
                return true;
            }

            bool EndArray(rapidjson::SizeType) {
                if (mode == Mode::NODES_ARRAY) {
                    mode = Mode::ROOT;
                } else if (mode == Mode::ATTR_VALS_ARRAY) {
                    mode = Mode::ATTR_VALS_OBJECT;
                } else if (mode == Mode::NODE_ARRAY) {
                    if (building_node) {
                        Eigen::VectorXf v(temp_array.size());
                        for (size_t i = 0; i < temp_array.size(); ++i) v(i) = temp_array[i];
                        if (array_key == "mean") building_node->mean = v;
                        else building_node->sum_sq = v;
                    }
                    mode = Mode::NODE_OBJECT;
                }
                return true;
            }

            bool Null() {
                if (mode == Mode::NODE_OBJECT && current_key == "parent_id") current_parent_id = -1;
                return true;
            }

            bool Bool(bool b) {
                if (mode == Mode::TREE_PARAMETERS && current_key == "weight_attr") tree->weight_attr = b;
                return true;
            }

            bool Int(int i) {
                if (mode == Mode::NODE_OBJECT) {
                    if (current_key == "id") current_id = i;
                    else if (current_key == "parent_id") current_parent_id = i;
                    else if (current_key == "count") node_count = i;
                } else if (mode == Mode::TREE_PARAMETERS) {
                    if (current_key == "size") tree->size = i;
                    else if (current_key == "depth_max") tree->depth_max = i;
                    else if (current_key == "branch_max") tree->branch_max = i;
                    else return Double(static_cast<double>(i));
                } else if (mode == Mode::ATTR_VALS_ARRAY) {
                    tree->attr_vals[current_map_key].insert(i);
                } else {
                    return Double(static_cast<double>(i));
                }
                return true;
            }
            bool Uint(unsigned u) { return Int(static_cast<int>(u)); }
            bool Int64(int64_t i) { return Int(static_cast<int>(i)); }
            bool Uint64(uint64_t u) { return Int(static_cast<int>(u)); }

            bool Double(double d) {
                switch (mode) {
                    case Mode::TREE_PARAMETERS:
                        if (current_key == "alpha") tree->alpha = static_cast<float>(d);
                        else if (current_key == "prior_var") tree->prior_var = static_cast<float>(d);
                        else if (current_key == "cont_weight") tree->cont_weight = static_cast<float>(d);
                        return true;
                    case Mode::ATTR_WEIGHTS_OBJECT:
                        tree->attr_weights[current_map_key] = d;
                        return true;
                    case Mode::NODE_OBJECT:
                        if (current_key == "count") node_count = d;
                        return true;
                    case Mode::NODE_ARRAY:
                        temp_array.push_back(static_cast<float>(d));
                        return true;
                    case Mode::A_COUNT_OBJECT:
                        if (!building_node) return false;
                        building_node->a_count[current_map_key] = d;
                        return true;
                    case Mode::SUM_N_LOGN_OBJECT:
                        if (!building_node) return false;
                        building_node->sum_n_logn[current_map_key] = d;
                        return true;
                    case Mode::AV_COUNT_INNER_OBJECT:
                        if (!building_node) return false;
                        building_node->av_count[current_attr_key][current_map_key] = d;
                        return true;
                    default:
                        return true;
                }
            }
    };

} // namespace

void CobwebHybridTree::load_json(const std::string &filename)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Reader reader;
    HybridTreeSaxHandler handler(this);

    if (!reader.Parse(isw, handler)) {
        throw std::runtime_error("SAX parse error while loading: " + filename);
    }
    if (this->root == nullptr) {
        throw std::runtime_error("No root node found while loading: " + filename);
    }

    for (const auto &[child_id, parent_id] : handler.child_to_parent) {
        auto *child = handler.id_to_node.at(child_id);
        auto *parent = handler.id_to_node.at(parent_id);
        child->parent = parent;
        parent->children.push_back(child);
    }
}


double CobwebHybridTree::rand()
{
    return this->own_rng ? this->unif(this->rng) : custom_rand();
}

void CobwebHybridTree::set_random_seed(uint64_t seed)
{
    this->own_rng = true;
    this->rng.seed(seed);
    this->unif.reset();
}
