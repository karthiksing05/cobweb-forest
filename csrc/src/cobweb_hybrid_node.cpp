#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "helper.h"
#include "cobweb_hybrid_node.h"
#include "cobweb_hybrid_tree.h"

// Global atomic counter to help guarantee unique concept ids
static std::atomic<uint64_t> HYBRID_CONCEPT_COUNTER{0};
static std::atomic<long long> HYBRID_UID_COUNTER{0};

static std::string generate_concept_hash()
{
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    uint64_t ns = duration_cast<nanoseconds>(now.time_since_epoch()).count();
    uint64_t id = ++HYBRID_CONCEPT_COUNTER;
    std::ostringstream ss;
    ss << ns << "_" << id;
    return ss.str();
}

// Closed-form Dirichlet-smoothed categorical entropy used by the discrete
// tree.  N = attribute count, K = number of known values, m = number of
// values with non-zero count in this node, snl = sum over those values of
// (n_v + alpha) log(n_v + alpha).  Values with zero count contribute
// alpha log alpha each (guarded so alpha == 0 does not produce 0 * -inf).
static double categorical_entropy(double N, int K, int m, double snl,
                                  double ratio, double alpha)
{
    int n0 = K - m;
    double zero_term = (alpha > 0.0 && n0 > 0) ? n0 * alpha * std::log(alpha) : 0.0;
    double denom = N + K * alpha;
    if (denom <= 0.0) return 0.0;
    return -ratio * ((snl + zero_term) / denom - std::log(denom));
}

// Differential entropy of a diagonal Gaussian with per-dimension variance
// ``var``: 0.5 * sum_d log(2 pi e var_d), scaled by ``weight``.  The
// constant cancels in every partition-utility difference, but keeping it
// makes ``entropy()`` a real (differential) entropy.
static double gaussian_entropy(const Eigen::ArrayXf &var, double weight)
{
    if (var.size() == 0) return 0.0;
    const double two_pi_e = 2.0 * M_PI * M_E;
    double h = 0.0;
    for (Eigen::Index d = 0; d < var.size(); ++d) {
        h += std::log(two_pi_e * static_cast<double>(var(d)));
    }
    return weight * 0.5 * h;
}

CobwebHybridNode::CobwebHybridNode(int size)
    : tree(nullptr),
    parent(nullptr),
    children(),
    count(0),
    a_count(),
    sum_n_logn(),
    av_count(),
    mean(Eigen::VectorXf::Zero(size)),
    sum_sq(Eigen::VectorXf::Zero(size))
{
    this->concept_id = generate_concept_hash();
    this->uid = ++HYBRID_UID_COUNTER;
}

CobwebHybridNode::CobwebHybridNode(CobwebHybridNode *other)
    : tree(other->tree),
    parent(other->parent),
    children(),
    count(0),
    a_count(),
    sum_n_logn(),
    av_count(),
    mean(Eigen::VectorXf::Zero(other->mean.size())),
    sum_sq(Eigen::VectorXf::Zero(other->sum_sq.size()))
{
    this->concept_id = generate_concept_hash();
    this->uid = ++HYBRID_UID_COUNTER;
    update_counts_from_node(other);
    for (auto *child : other->children) {
        auto *copy = new CobwebHybridNode(child);
        copy->parent = this;
        children.push_back(copy);
    }
}

CobwebHybridNode::~CobwebHybridNode() {
    for (auto *child : children) {
        delete child;
    }
}

// ============================================================
// Count maintenance
// ============================================================

void CobwebHybridNode::increment_counts(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont)
{
    const double alpha = this->tree->alpha;

    this->count += 1;

    for (auto &[attr, val_map] : disc) {
        for (auto &[val, cnt] : val_map) {
            this->a_count[attr] += cnt;

            // only maintain sum_n_logn for non-hidden attrs (non negative)
            if (attr >= 0) {
                auto ait = this->av_count.find(attr);
                if (ait != this->av_count.end()) {
                    auto vit = ait->second.find(val);
                    if (vit != ait->second.end()) {
                        double tf = vit->second + alpha;
                        this->sum_n_logn[attr] -= tf * std::log(tf);
                    }
                }
            }

            this->av_count[attr][val] += cnt;

            if (attr >= 0) {
                double tf = this->av_count.at(attr).at(val) + alpha;
                this->sum_n_logn[attr] += tf * std::log(tf);
            }
        }
    }

    if (this->tree->size > 0) {
        if (cont.size() != this->tree->size) {
            throw std::invalid_argument(
                "continuous vector has size " + std::to_string(cont.size()) +
                " but the tree expects " + std::to_string(this->tree->size));
        }
        Eigen::VectorXf delta = cont - this->mean;
        this->mean += delta / static_cast<float>(this->count);
        this->sum_sq += delta.cwiseProduct(cont - this->mean);
    }
}

void CobwebHybridNode::update_counts_from_node(CobwebHybridNode *other)
{
    const double alpha = this->tree->alpha;

    // Continuous block first: Chan et al. parallel merge of two Welford
    // accumulators needs both counts before they are summed.
    if (this->tree->size > 0 && other->count > 0) {
        if (this->count > 0) {
            float n1 = static_cast<float>(this->count);
            float n2 = static_cast<float>(other->count);
            Eigen::VectorXf delta = other->mean - this->mean;
            this->sum_sq = (this->sum_sq + other->sum_sq +
                            delta.cwiseProduct(delta) * ((n1 * n2) / (n1 + n2)));
            this->mean = (n1 * this->mean + n2 * other->mean) / (n1 + n2);
        } else {
            this->mean = other->mean;
            this->sum_sq = other->sum_sq;
        }
    }

    this->count += other->count;

    for (auto &[attr, val_map] : other->av_count) {
        this->a_count[attr] += other->a_count.at(attr);

        for (auto &[val, cnt] : val_map) {
            if (attr >= 0) {
                auto ait = this->av_count.find(attr);
                if (ait != this->av_count.end()) {
                    auto vit = ait->second.find(val);
                    if (vit != ait->second.end()) {
                        double tf = vit->second + alpha;
                        this->sum_n_logn[attr] -= tf * std::log(tf);
                    }
                }
            }

            this->av_count[attr][val] += cnt;

            if (attr >= 0) {
                double tf = this->av_count.at(attr).at(val) + alpha;
                this->sum_n_logn[attr] += tf * std::log(tf);
            }
        }
    }
}

// ============================================================
// Discrete entropies
// ============================================================

double CobwebHybridNode::attr_ratio(ATTR_TYPE attr)
{
    // Attributes that carry more than one count per instance (e.g. bags of
    // context words) act like several attributes; weight them accordingly.
    double ratio = 1.0;
    if (this->tree->weight_attr && this->tree->root->count > 0) {
        auto rit = this->tree->root->a_count.find(attr);
        if (rit != this->tree->root->a_count.end()) {
            ratio = rit->second / this->tree->root->count;
        }
    }
    auto wit = this->tree->attr_weights.find(attr);
    if (wit != this->tree->attr_weights.end()) {
        ratio *= wit->second;
    }
    return ratio;
}

double CobwebHybridNode::entropy_attr(ATTR_TYPE attr)
{
    // hidden attributes (negative) have zero entropy.
    if (attr < 0) return 0.0;

    auto kit = this->tree->attr_vals.find(attr);
    if (kit == this->tree->attr_vals.end()) return 0.0;
    const int K = static_cast<int>(kit->second.size());
    const double alpha = this->tree->alpha;

    double N = 0.0;
    int m = 0;
    auto ait = this->av_count.find(attr);
    if (ait != this->av_count.end()) {
        N = this->a_count.at(attr);
        m = static_cast<int>(ait->second.size());
    }

    // No observations and no smoothing: the distribution is undefined; treat
    // the entropy as 0 (same convention as the discrete tree).
    if (N == 0.0 && alpha == 0.0) return 0.0;

    double snl = 0.0;
    auto sit = this->sum_n_logn.find(attr);
    if (sit != this->sum_n_logn.end()) snl = sit->second;

    return categorical_entropy(N, K, m, snl, this->attr_ratio(attr), alpha);
}

double CobwebHybridNode::entropy_attr_insert(ATTR_TYPE attr, const INSTANCE_TYPE &disc)
{
    if (attr < 0) return 0.0;

    auto kit = this->tree->attr_vals.find(attr);
    if (kit == this->tree->attr_vals.end()) return 0.0;
    const int K = static_cast<int>(kit->second.size());
    const double alpha = this->tree->alpha;

    double N = 0.0;
    int m = 0;
    auto ait = this->av_count.find(attr);
    if (ait != this->av_count.end()) {
        N = this->a_count.at(attr);
        m = static_cast<int>(ait->second.size());
    }

    double snl = 0.0;
    auto sit = this->sum_n_logn.find(attr);
    if (sit != this->sum_n_logn.end()) snl = sit->second;

    auto dit = disc.find(attr);
    if (dit != disc.end()) {
        for (auto &[val, cnt] : dit->second) {
            N += cnt;
            double prior = 0.0;
            if (ait != this->av_count.end()) {
                auto vit = ait->second.find(val);
                if (vit != ait->second.end()) {
                    prior = vit->second;
                    double tf = prior + alpha;
                    snl -= tf * std::log(tf);
                } else {
                    m += 1;
                }
            } else {
                m += 1;
            }
            double tf = prior + cnt + alpha;
            snl += tf * std::log(tf);
        }
    }

    if (N == 0.0 && alpha == 0.0) return 0.0;
    return categorical_entropy(N, K, m, snl, this->attr_ratio(attr), alpha);
}

double CobwebHybridNode::entropy_attr_merge(ATTR_TYPE attr, CobwebHybridNode *other,
        const INSTANCE_TYPE &disc)
{
    if (attr < 0) return 0.0;

    auto kit = this->tree->attr_vals.find(attr);
    if (kit == this->tree->attr_vals.end()) return 0.0;
    const int K = static_cast<int>(kit->second.size());
    const double alpha = this->tree->alpha;

    double N = 0.0;
    int m = 0;
    auto ait = this->av_count.find(attr);
    if (ait != this->av_count.end()) {
        N = this->a_count.at(attr);
        m = static_cast<int>(ait->second.size());
    }

    double snl = 0.0;
    auto sit = this->sum_n_logn.find(attr);
    if (sit != this->sum_n_logn.end()) snl = sit->second;

    auto oit = other->av_count.find(attr);
    auto dit = disc.find(attr);

    // Fold in ``other`` (plus the instance's count for the same value).
    if (oit != other->av_count.end()) {
        for (auto &[val, other_cnt] : oit->second) {
            double inst_cnt = 0.0;
            if (dit != disc.end()) {
                auto ivit = dit->second.find(val);
                if (ivit != dit->second.end()) inst_cnt = ivit->second;
            }
            N += other_cnt + inst_cnt;
            double prior = 0.0;
            if (ait != this->av_count.end()) {
                auto vit = ait->second.find(val);
                if (vit != ait->second.end()) {
                    prior = vit->second;
                    double tf = prior + alpha;
                    snl -= tf * std::log(tf);
                } else {
                    m += 1;
                }
            } else {
                m += 1;
            }
            double tf = prior + other_cnt + inst_cnt + alpha;
            snl += tf * std::log(tf);
        }
    }

    // Fold in instance values that ``other`` does not have.
    if (dit != disc.end()) {
        for (auto &[val, inst_cnt] : dit->second) {
            if (oit != other->av_count.end() && oit->second.count(val)) continue;
            N += inst_cnt;
            double prior = 0.0;
            if (ait != this->av_count.end()) {
                auto vit = ait->second.find(val);
                if (vit != ait->second.end()) {
                    prior = vit->second;
                    double tf = prior + alpha;
                    snl -= tf * std::log(tf);
                } else {
                    m += 1;
                }
            } else {
                m += 1;
            }
            double tf = prior + inst_cnt + alpha;
            snl += tf * std::log(tf);
        }
    }

    if (N == 0.0 && alpha == 0.0) return 0.0;
    return categorical_entropy(N, K, m, snl, this->attr_ratio(attr), alpha);
}

double CobwebHybridNode::entropy_discrete()
{
    double info = 0.0;
    for (auto &[attr, val_set] : this->tree->attr_vals) {
        if (attr < 0) continue;
        info += this->entropy_attr(attr);
    }
    return info;
}

double CobwebHybridNode::entropy_discrete_insert(const INSTANCE_TYPE &disc)
{
    double info = 0.0;
    for (auto &[attr, val_set] : this->tree->attr_vals) {
        if (attr < 0) continue;
        info += this->entropy_attr_insert(attr, disc);
    }
    return info;
}

double CobwebHybridNode::entropy_discrete_merge(CobwebHybridNode *other, const INSTANCE_TYPE &disc)
{
    double info = 0.0;
    for (auto &[attr, val_set] : this->tree->attr_vals) {
        if (attr < 0) continue;
        info += this->entropy_attr_merge(attr, other, disc);
    }
    return info;
}

// ============================================================
// Continuous entropies
// ============================================================

Eigen::ArrayXf CobwebHybridNode::model_variance(double eval_prior_var) const
{
    float pv = (eval_prior_var >= 0.0) ? static_cast<float>(eval_prior_var)
                                       : this->tree->prior_var;
    if (this->count > 0) {
        return this->sum_sq.array() / static_cast<float>(this->count) + pv;
    }
    return Eigen::ArrayXf::Constant(this->mean.size(), pv);
}

Eigen::ArrayXf CobwebHybridNode::variance_insert(const Eigen::VectorXf &cont) const
{
    float n = static_cast<float>(this->count) + 1.0f;
    Eigen::VectorXf delta = cont - this->mean;
    Eigen::VectorXf new_mean = this->mean + delta / n;
    Eigen::VectorXf new_sum_sq = this->sum_sq + delta.cwiseProduct(cont - new_mean);
    return new_sum_sq.array() / n + this->tree->prior_var;
}

Eigen::ArrayXf CobwebHybridNode::variance_merge(CobwebHybridNode *other,
        const Eigen::VectorXf &cont) const
{
    float n1 = static_cast<float>(this->count);
    float n2 = static_cast<float>(other->count);
    Eigen::VectorXf m;
    Eigen::VectorXf ss;
    if (n1 + n2 > 0.0f) {
        Eigen::VectorXf delta = other->mean - this->mean;
        ss = this->sum_sq + other->sum_sq + delta.cwiseProduct(delta) * ((n1 * n2) / (n1 + n2));
        m = (n1 * this->mean + n2 * other->mean) / (n1 + n2);
    } else {
        ss = Eigen::VectorXf::Zero(this->mean.size());
        m = Eigen::VectorXf::Zero(this->mean.size());
    }
    float n = n1 + n2;
    if (cont.size() > 0) {
        n += 1.0f;
        Eigen::VectorXf delta = cont - m;
        m += delta / n;
        ss += delta.cwiseProduct(cont - m);
    }
    if (n <= 0.0f) return Eigen::ArrayXf::Constant(this->mean.size(), this->tree->prior_var);
    return ss.array() / n + this->tree->prior_var;
}

double CobwebHybridNode::entropy_continuous()
{
    if (this->tree->size == 0) return 0.0;
    return gaussian_entropy(this->model_variance(), this->tree->cont_weight);
}

double CobwebHybridNode::entropy_continuous_insert(const Eigen::VectorXf &cont)
{
    if (this->tree->size == 0) return 0.0;
    // Missing continuous value: nothing to add, the block is unchanged.
    if (cont.size() == 0) return this->entropy_continuous();
    return gaussian_entropy(this->variance_insert(cont), this->tree->cont_weight);
}

double CobwebHybridNode::entropy_continuous_merge(CobwebHybridNode *other,
        const Eigen::VectorXf &cont)
{
    if (this->tree->size == 0) return 0.0;
    return gaussian_entropy(this->variance_merge(other, cont), this->tree->cont_weight);
}

double CobwebHybridNode::entropy_continuous_new(const Eigen::VectorXf &cont)
{
    if (this->tree->size == 0) return 0.0;
    (void)cont;  // a fresh singleton concept has variance prior_var regardless of x
    Eigen::ArrayXf var = Eigen::ArrayXf::Constant(this->tree->size, this->tree->prior_var);
    return gaussian_entropy(var, this->tree->cont_weight);
}

// ============================================================
// Totals
// ============================================================

double CobwebHybridNode::entropy()
{
    return this->entropy_discrete() + this->entropy_continuous();
}

double CobwebHybridNode::entropy_insert(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont)
{
    return this->entropy_discrete_insert(disc) + this->entropy_continuous_insert(cont);
}

double CobwebHybridNode::entropy_merge(CobwebHybridNode *other, const INSTANCE_TYPE &disc,
        const Eigen::VectorXf &cont)
{
    return this->entropy_discrete_merge(other, disc) + this->entropy_continuous_merge(other, cont);
}

// ============================================================
// Partition utility
// ============================================================

double CobwebHybridNode::partition_utility()
{
    if (this->children.empty()) return 0.0;

    double children_entropy = 0.0;
    for (auto *child : this->children) {
        double p_of_child = (1.0 * child->count) / this->count;
        children_entropy += p_of_child * child->entropy();
    }
    return (this->entropy() - children_entropy) / this->children.size();
}

double CobwebHybridNode::pu_for_insert(CobwebHybridNode *child, const INSTANCE_TYPE &disc,
        const Eigen::VectorXf &cont)
{
    double children_entropy = 0.0;
    for (auto *c : this->children) {
        if (c == child) {
            double p_of_child = (c->count + 1.0) / (this->count + 1.0);
            children_entropy += p_of_child * c->entropy_insert(disc, cont);
        } else {
            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            children_entropy += p_of_child * c->entropy();
        }
    }
    double parent_entropy = this->entropy_insert(disc, cont);
    return (parent_entropy - children_entropy) / this->children.size();
}

double CobwebHybridNode::pu_for_new_child(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont)
{
    CobwebHybridNode new_child(this->tree->size);
    new_child.parent = this;
    new_child.tree = this->tree;
    new_child.increment_counts(disc, cont);
    double p_of_new_child = 1.0 / (this->count + 1.0);

    double children_entropy = p_of_new_child * new_child.entropy();
    for (auto *c : this->children) {
        double p_of_child = (1.0 * c->count) / (this->count + 1.0);
        children_entropy += p_of_child * c->entropy();
    }
    double parent_entropy = this->entropy_insert(disc, cont);
    return (parent_entropy - children_entropy) / (this->children.size() + 1);
}

double CobwebHybridNode::pu_for_merge(CobwebHybridNode *best1, CobwebHybridNode *best2,
        const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont)
{
    double children_entropy = 0.0;
    for (auto *c : this->children) {
        if (c == best1 || c == best2) continue;
        double p_of_child = (1.0 * c->count) / (this->count + 1.0);
        children_entropy += p_of_child * c->entropy();
    }
    double p_of_child = (best1->count + best2->count + 1.0) / (this->count + 1.0);
    children_entropy += p_of_child * best1->entropy_merge(best2, disc, cont);

    double parent_entropy = this->entropy_insert(disc, cont);
    return (parent_entropy - children_entropy) / (this->children.size() - 1);
}

double CobwebHybridNode::pu_for_split(CobwebHybridNode *best)
{
    double children_entropy = 0.0;
    for (auto *c : this->children) {
        if (c == best) continue;
        double p_of_child = (1.0 * c->count) / this->count;
        children_entropy += p_of_child * c->entropy();
    }
    for (auto *c : best->children) {
        double p_of_child = (1.0 * c->count) / this->count;
        children_entropy += p_of_child * c->entropy();
    }
    double parent_entropy = this->entropy();
    return (parent_entropy - children_entropy) /
           (this->children.size() - 1 + best->children.size());
}

std::tuple<double, CobwebHybridNode *, CobwebHybridNode *>
CobwebHybridNode::two_best_children(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont)
{
    if (this->children.empty()) {
        throw std::runtime_error("two_best_children called on a node with no children");
    }

    // Relative partition utility: only the candidate child's term changes,
    // so rank children by the change in (count * entropy) that adding the
    // instance to them causes.
    std::vector<std::tuple<double, double, double, CobwebHybridNode *>> relative_pu;
    relative_pu.reserve(this->children.size());
    for (auto *child : this->children) {
        relative_pu.push_back(std::make_tuple(
            (child->count * child->entropy()) -
                ((child->count + 1) * child->entropy_insert(disc, cont)),
            child->count,
            this->tree->rand(),
            child));
    }

    std::sort(relative_pu.rbegin(), relative_pu.rend());
    CobwebHybridNode *best1 = std::get<3>(relative_pu[0]);
    double best1_pu = this->pu_for_insert(best1, disc, cont);
    CobwebHybridNode *best2 = relative_pu.size() > 1 ? std::get<3>(relative_pu[1]) : nullptr;
    return std::make_tuple(best1_pu, best1, best2);
}

OPERATION_TYPE CobwebHybridNode::get_best_operation(const INSTANCE_TYPE &disc,
        const Eigen::VectorXf &cont, CobwebHybridNode *best1,
        CobwebHybridNode *best2, double best1_pu)
{
    if (best1 == nullptr) {
        throw std::runtime_error("get_best_operation needs at least one best child");
    }

    std::vector<std::tuple<double, double, Operation>> operations;
    operations.push_back(std::make_tuple(best1_pu, this->tree->rand(), Operation::BEST));

    if (this->tree->add_only && !this->tree->allow_new) {
        return std::make_pair(best1_pu, Operation::BEST);
    }

    if (this->children.size() < (size_t)this->tree->branch_max) {
        operations.push_back(std::make_tuple(this->pu_for_new_child(disc, cont),
                    this->tree->rand(), Operation::NEW));
        if (this->tree->add_only) {
            // add_only + allow_new: BEST or NEW only.  Neither deletes or
            // re-parents a node, so stored paths / symbols stay valid, but
            // the tree can grow wide where the utility says so instead of
            // only through binary fringe splits (which form chains).
            std::sort(operations.rbegin(), operations.rend());
            return std::make_pair(std::get<0>(operations[0]), std::get<2>(operations[0]));
        }

        if (this->children.size() > 2 && best2 != nullptr) {
            operations.push_back(std::make_tuple(this->pu_for_merge(best1, best2, disc, cont),
                        this->tree->rand(), Operation::MERGE));
        }

        if (best1->children.size() > 1 &&
                (this->children.size() - 1 + best1->children.size()) <= (size_t)this->tree->branch_max) {
            operations.push_back(std::make_tuple(this->pu_for_split(best1),
                        this->tree->rand(), Operation::SPLIT));
        }
    }

    std::sort(operations.rbegin(), operations.rend());
    return std::make_pair(std::get<0>(operations[0]), std::get<2>(operations[0]));
}

bool CobwebHybridNode::is_exact_match(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont)
{
    // --- discrete: same attributes, same value distribution ---
    std::unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp] : disc) all_attrs.insert(attr);
    for (auto &[attr, tmp] : this->av_count) all_attrs.insert(attr);

    for (auto attr : all_attrs) {
        if (attr < 0) continue;  // hidden attributes are ignored

        auto dit = disc.find(attr);
        auto ait = this->av_count.find(attr);
        bool in_inst = (dit != disc.end());
        bool in_concept = (ait != this->av_count.end());
        if (in_inst != in_concept) return false;
        if (!in_inst) continue;

        double instance_attr_count = 0.0;
        for (auto &[val, cnt] : dit->second) instance_attr_count += cnt;

        std::unordered_set<VALUE_TYPE> all_vals;
        for (auto &[val, cnt] : dit->second) all_vals.insert(val);
        for (auto &[val, cnt] : ait->second) all_vals.insert(val);

        for (auto val : all_vals) {
            auto ivit = dit->second.find(val);
            auto cvit = ait->second.find(val);
            if ((ivit == dit->second.end()) != (cvit == ait->second.end())) return false;
            double instance_prob = ivit->second / instance_attr_count;
            double concept_prob = cvit->second / this->a_count.at(attr);
            if (std::abs(instance_prob - concept_prob) > 1e-5) return false;
        }
    }

    // --- continuous: zero spread and the instance sits on the mean ---
    if (this->tree->size > 0 && cont.size() > 0) {
        if (this->count > 0) {
            Eigen::VectorXf var = this->sum_sq / static_cast<float>(this->count);
            if (!is_close_to_zero(var, 1e-5f)) return false;
        }
        Eigen::VectorXf delta = cont - this->mean;
        if (!is_close_to_zero(delta, 1e-5f)) return false;
    }

    return true;
}

// ============================================================
// Probabilities
// ============================================================

double CobwebHybridNode::log_prob_instance(const INSTANCE_TYPE &disc, const Eigen::VectorXf &cont,
        double eval_alpha, double eval_prior_var)
{
    double eff_alpha = (eval_alpha < 0.0) ? this->tree->alpha : eval_alpha;
    double log_prob = 0.0;

    for (auto &[attr, val_map] : disc) {
        if (attr < 0) continue;
        auto kit = this->tree->attr_vals.find(attr);
        if (kit == this->tree->attr_vals.end()) continue;

        double num_vals = static_cast<double>(kit->second.size());
        double attr_count = (num_vals + 1) * eff_alpha;  // one extra slot for "unseen"
        auto acit = this->a_count.find(attr);
        if (acit != this->a_count.end()) attr_count += acit->second;

        auto ait = this->av_count.find(attr);
        for (auto &[val, cnt] : val_map) {
            double av = eff_alpha;
            if (kit->second.count(val) && ait != this->av_count.end()) {
                auto vit = ait->second.find(val);
                if (vit != ait->second.end()) av += vit->second;
            }
            log_prob += cnt * (std::log(av) - std::log(attr_count));
        }
    }

    if (this->tree->size > 0 && cont.size() > 0) {
        // forest iteration: NaN entries of ``cont`` are missing dimensions and
        // are not scored (the marginal of a diagonal Gaussian drops them).
        Eigen::ArrayXf var = this->model_variance(eval_prior_var);
        const double log_2pi = std::log(2.0 * M_PI);
        for (Eigen::Index d = 0; d < cont.size(); ++d) {
            float x = cont(d);
            if (std::isnan(x)) continue;
            double v = static_cast<double>(var(d));
            double delta = static_cast<double>(x) - static_cast<double>(this->mean(d));
            log_prob += -0.5 * (std::log(v) + log_2pi + delta * delta / v);
        }
    }

    return log_prob;
}

double CobwebHybridNode::log_prob_class_given_instance(const INSTANCE_TYPE &disc,
        const Eigen::VectorXf &cont)
{
    double log_prob = this->log_prob_instance(disc, cont);
    log_prob += std::log((1.0 * this->count) / this->tree->root->count);
    return log_prob;
}

std::vector<double> CobwebHybridNode::log_prob_children_given_instance(const INSTANCE_TYPE &disc,
        const Eigen::VectorXf &cont)
{
    std::vector<double> raw_log_probs;
    std::vector<double> norm_log_probs;
    for (auto *child : this->children) {
        raw_log_probs.push_back(child->log_prob_class_given_instance(disc, cont));
    }
    double log_p_of_x = logsumexp(raw_log_probs);
    for (auto log_p : raw_log_probs) {
        norm_log_probs.push_back(log_p - log_p_of_x);
    }
    return norm_log_probs;
}

INSTANCE_TYPE CobwebHybridNode::predict_probs()
{
    INSTANCE_TYPE out;
    const double alpha = this->tree->alpha;
    for (auto &[attr, val_set] : this->tree->attr_vals) {
        if (attr < 0) continue;
        double num_vals = static_cast<double>(val_set.size());
        double attr_count = 0.0;
        auto acit = this->a_count.find(attr);
        if (acit != this->a_count.end()) attr_count = acit->second;
        auto ait = this->av_count.find(attr);
        for (auto val : val_set) {
            double av = 0.0;
            if (ait != this->av_count.end()) {
                auto vit = ait->second.find(val);
                if (vit != ait->second.end()) av = vit->second;
            }
            out[attr][val] = (av + alpha) / (attr_count + num_vals * alpha);
        }
    }
    return out;
}

double CobwebHybridNode::prob(ATTR_TYPE attr, VALUE_TYPE val)
{
    auto kit = this->tree->attr_vals.find(attr);
    if (kit == this->tree->attr_vals.end()) return 0.0;
    double num_vals = static_cast<double>(kit->second.size());
    double attr_count = 0.0;
    auto acit = this->a_count.find(attr);
    if (acit != this->a_count.end()) attr_count = acit->second;
    double av = 0.0;
    auto ait = this->av_count.find(attr);
    if (ait != this->av_count.end()) {
        auto vit = ait->second.find(val);
        if (vit != ait->second.end()) av = vit->second;
    }
    return (av + this->tree->alpha) / (attr_count + num_vals * this->tree->alpha);
}

// ============================================================
// Basic level: closed-form expected PMI against the root
// ============================================================

double CobwebHybridNode::expected_pmi_discrete(double eval_alpha)
{
    /*
     * E_{x ~ c}[ log P_c(x) - log P_root(x) ] for the discrete block.  The
     * expectation is over the empirical distribution of the instances stored
     * under c, which (because av_count aggregates the leaves) is just c's own
     * relative frequencies:
     *
     *   sum_a (1 / N_c^a) sum_v n_v^c [ log P_c(v|a) - log P_root(v|a) ]
     *
     * with P(v|a) = (n_v + alpha) / (N + K_a alpha).  This equals the
     * all-leaves ``use_root`` estimator of CobwebDiscreteNode::expected_pmi.
     */
    CobwebHybridNode *root = this->tree->root;
    if (this == root) return 0.0;
    double a = (eval_alpha < 0.0) ? this->tree->alpha : eval_alpha;

    double total = 0.0;
    for (auto &[attr, val_set] : this->tree->attr_vals) {
        if (attr < 0) continue;
        auto ait = this->av_count.find(attr);
        if (ait == this->av_count.end()) continue;
        double K = static_cast<double>(val_set.size());
        double N_c = this->a_count.at(attr);
        if (N_c <= 0.0) continue;
        double N_r = root->a_count.count(attr) ? root->a_count.at(attr) : 0.0;
        double denom_c = N_c + K * a;
        double denom_r = N_r + K * a;
        auto rit = root->av_count.find(attr);

        double attr_total = 0.0;
        for (auto &[val, n_v_c] : ait->second) {
            if (n_v_c <= 0.0) continue;
            double n_v_r = 0.0;
            if (rit != root->av_count.end()) {
                auto vit = rit->second.find(val);
                if (vit != rit->second.end()) n_v_r = vit->second;
            }
            double log_P_c = std::log((n_v_c + a) / denom_c);
            double log_P_r = std::log((n_v_r + a) / denom_r);
            attr_total += n_v_c * (log_P_c - log_P_r);
        }
        total += attr_total / N_c;
    }
    return total;
}

double CobwebHybridNode::expected_pmi_continuous(double eval_prior_var)
{
    /*
     * Closed form of E_{x ~ N(mu_c, s2_c)}[ log p_c(x) - log p_root(x) ] for
     * diagonal Gaussians (see CobwebContinuousNode::expected_pmi):
     *
     *   -0.5 * sum_d [ log(v_c/v_r) + s2_c/v_c - (s2_c + (mu_c - mu_r)^2)/v_r ]
     *
     * where s2 = sum_sq / count is the empirical variance and v = s2 + prior_var
     * the regularised model variance.
     */
    if (this->tree->size == 0) return 0.0;
    CobwebHybridNode *root = this->tree->root;
    if (this == root) return 0.0;
    if (this->count <= 0 || root->count <= 0) return 0.0;

    float pv = (eval_prior_var >= 0.0) ? static_cast<float>(eval_prior_var)
                                       : this->tree->prior_var;

    Eigen::ArrayXf s2_c = this->sum_sq.array() / static_cast<float>(this->count);
    Eigen::ArrayXf v_c  = s2_c + pv;
    Eigen::ArrayXf s2_r = root->sum_sq.array() / static_cast<float>(root->count);
    Eigen::ArrayXf v_r  = s2_r + pv;
    Eigen::ArrayXf dmu  = (this->mean - root->mean).array();

    Eigen::ArrayXf term = (v_c / v_r).log() + (s2_c / v_c) - ((s2_c + dmu.square()) / v_r);
    return -0.5 * static_cast<double>(term.sum());
}

double CobwebHybridNode::expected_pmi(double eval_alpha, double eval_prior_var)
{
    return this->expected_pmi_discrete(eval_alpha) + this->expected_pmi_continuous(eval_prior_var);
}

CobwebHybridNode *CobwebHybridNode::get_basic(double eval_alpha, double eval_prior_var)
{
    // Walk from this node to the root and return the node with the highest
    // expected PMI (the "basic level" for this branch).
    CobwebHybridNode *curr = this;
    CobwebHybridNode *best = this;
    double best_epmi = this->expected_pmi(eval_alpha, eval_prior_var);

    while (curr->parent != nullptr) {
        curr = curr->parent;
        double curr_epmi = curr->expected_pmi(eval_alpha, eval_prior_var);
        if (curr_epmi > best_epmi) {
            best = curr;
            best_epmi = curr_epmi;
        }
    }
    return best;
}

// ============================================================
// Misc
// ============================================================

std::size_t CobwebHybridNode::_hash()
{
    return std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(this));
}

std::string CobwebHybridNode::concept_hash()
{
    return this->concept_id;
}

int CobwebHybridNode::depth()
{
    int d = 0;
    CobwebHybridNode *p = this->parent;
    while (p != nullptr) {
        d++;
        p = p->parent;
    }
    return d;
}

int CobwebHybridNode::num_concepts()
{
    int n = 1;
    for (auto *c : this->children) n += c->num_concepts();
    return n;
}

bool CobwebHybridNode::is_parent(CobwebHybridNode *other)
{
    CobwebHybridNode *temp = other;
    while (temp != nullptr) {
        if (temp == this) return true;
        temp = temp->parent;
    }
    return false;
}
