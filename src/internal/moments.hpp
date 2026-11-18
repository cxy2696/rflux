#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace rflux::detail {

/*
 * Stable mergeable moments for a sliding window.
 *
 * Raw sum/sum-of-squares updates cancel catastrophically when the level is
 * large relative to the dispersion (prices around 1e9 with a 1e-2 range, for
 * example).  The two-stack queue below merges Welford-style block moments, so
 * every combine step uses centred differences.  Each push/pop is amortised
 * O(1) and the memory use is O(window).
 *
 * The queue is generic over the moment algebra; Moments1Ops and Moments2Ops
 * only supply the single-observation and block-combine operations.  Keeping
 * the queue in one place means the transfer/pop/aggregate invariants are
 * written and tested once instead of once per arity.
 */

struct Moments1 {
    long double count = 0.0L;
    long double mean = 0.0L;
    long double m2 = 0.0L;
};

inline Moments1 single_moment(double value) {
    return Moments1{1.0L, static_cast<long double>(value), 0.0L};
}

/* Shared count/inverse/weight arithmetic for both moment arities. */
struct MergeScale {
    long double count;
    long double inverse;
    long double weight;
};

inline MergeScale merge_scale(long double count_a, long double count_b) {
    const long double count = count_a + count_b;
    const long double inverse = 1.0L / count;
    return {count, inverse, count_a * count_b * inverse};
}

inline Moments1 merge(const Moments1& a, const Moments1& b) {
    if (a.count == 0.0L) return b;
    if (b.count == 0.0L) return a;
    const MergeScale scale = merge_scale(a.count, b.count);
    const long double delta = b.mean - a.mean;
    Moments1 result;
    result.count = scale.count;
    result.mean = a.mean + delta * (b.count * scale.inverse);
    result.m2 = a.m2 + b.m2 + delta * delta * scale.weight;
    return result;
}

/*
 * Append one observation to an existing block.  This is the hot path of a
 * sliding push: the second operand is always a singleton, so its count is one
 * and the general merge collapses to a single reciprocal.
 */
inline Moments1 merge_value(const Moments1& a, double value) {
    if (a.count == 0.0L) return single_moment(value);
    const long double count = a.count + 1.0L;
    const long double delta = static_cast<long double>(value) - a.mean;
    const long double inverse = 1.0L / count;
    Moments1 result;
    result.count = count;
    result.mean = a.mean + delta * inverse;
    result.m2 = a.m2 + delta * delta * (a.count * inverse);
    return result;
}

struct Moments1Ops {
    using Entry = double;
    using Aggregate = Moments1;

    static Aggregate single(const Entry& value) { return single_moment(value); }
    static Aggregate combine(const Aggregate& a, const Aggregate& b) {
        return merge(a, b);
    }
    static Aggregate append(const Aggregate& a, const Entry& value) {
        return merge_value(a, value);
    }
};

template <typename Ops>
class SlidingBlocks {
public:
    using Entry = typename Ops::Entry;
    using Aggregate = typename Ops::Aggregate;

    void push(const Entry& entry) {
        Node node{entry, {}};
        node.aggregate = back_.empty()
                             ? Ops::single(entry)
                             : Ops::append(back_.back().aggregate, entry);
        back_.push_back(node);
        ++size_;
    }

    void pop() {
        if (front_.empty()) transfer();
        front_.pop_back();
        --size_;
    }

    std::size_t size() const { return size_; }

    Aggregate aggregate() const {
        if (front_.empty()) {
            return back_.empty() ? Aggregate{} : back_.back().aggregate;
        }
        if (back_.empty()) return front_.back().aggregate;
        return Ops::combine(front_.back().aggregate, back_.back().aggregate);
    }

private:
    struct Node {
        Entry entry;
        Aggregate aggregate;
    };

    void transfer() {
        while (!back_.empty()) {
            Node node = back_.back();
            back_.pop_back();
            node.aggregate = Ops::single(node.entry);
            if (!front_.empty()) {
                node.aggregate =
                    Ops::combine(node.aggregate, front_.back().aggregate);
            }
            front_.push_back(node);
        }
    }

    std::vector<Node> back_;
    std::vector<Node> front_;
    std::size_t size_ = 0;
};

class SlidingMoments1 {
public:
    void push(double value) { blocks_.push(value); }
    void pop() { blocks_.pop(); }
    std::size_t size() const { return blocks_.size(); }
    Moments1 moments() const { return blocks_.aggregate(); }

private:
    SlidingBlocks<Moments1Ops> blocks_;
};

struct Moments2 {
    long double count = 0.0L;
    long double mean_x = 0.0L;
    long double mean_y = 0.0L;
    long double m2_x = 0.0L;
    long double m2_y = 0.0L;
    long double co_moment = 0.0L;
};

inline Moments2 single_moment2(double x, double y) {
    return Moments2{1.0L, static_cast<long double>(x),
                    static_cast<long double>(y), 0.0L, 0.0L, 0.0L};
}

inline Moments2 merge(const Moments2& a, const Moments2& b) {
    if (a.count == 0.0L) return b;
    if (b.count == 0.0L) return a;
    const MergeScale scale = merge_scale(a.count, b.count);
    const long double dx = b.mean_x - a.mean_x;
    const long double dy = b.mean_y - a.mean_y;
    Moments2 result;
    result.count = scale.count;
    result.mean_x = a.mean_x + dx * (b.count * scale.inverse);
    result.mean_y = a.mean_y + dy * (b.count * scale.inverse);
    result.m2_x = a.m2_x + b.m2_x + dx * dx * scale.weight;
    result.m2_y = a.m2_y + b.m2_y + dy * dy * scale.weight;
    result.co_moment = a.co_moment + b.co_moment + dx * dy * scale.weight;
    return result;
}

/* Singleton append for the two-variable sliding window; see merge_value. */
inline Moments2 merge_pair(const Moments2& a, double x, double y) {
    if (a.count == 0.0L) return single_moment2(x, y);
    const long double count = a.count + 1.0L;
    const long double dx = static_cast<long double>(x) - a.mean_x;
    const long double dy = static_cast<long double>(y) - a.mean_y;
    const long double inverse = 1.0L / count;
    const long double weight = a.count * inverse;
    Moments2 result;
    result.count = count;
    result.mean_x = a.mean_x + dx * inverse;
    result.mean_y = a.mean_y + dy * inverse;
    result.m2_x = a.m2_x + dx * dx * weight;
    result.m2_y = a.m2_y + dy * dy * weight;
    result.co_moment = a.co_moment + dx * dy * weight;
    return result;
}

struct Moments2Entry {
    double x;
    double y;
};

struct Moments2Ops {
    using Entry = Moments2Entry;
    using Aggregate = Moments2;

    static Aggregate single(const Entry& entry) {
        return single_moment2(entry.x, entry.y);
    }
    static Aggregate combine(const Aggregate& a, const Aggregate& b) {
        return merge(a, b);
    }
    static Aggregate append(const Aggregate& a, const Entry& entry) {
        return merge_pair(a, entry.x, entry.y);
    }
};

class SlidingMoments2 {
public:
    void push(double x, double y) { blocks_.push(Moments2Entry{x, y}); }
    void pop() { blocks_.pop(); }
    std::size_t size() const { return blocks_.size(); }
    Moments2 moments() const { return blocks_.aggregate(); }

private:
    SlidingBlocks<Moments2Ops> blocks_;
};

}  // namespace rflux::detail
