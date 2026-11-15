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
 */

struct Moments1 {
    long double count = 0.0L;
    long double mean = 0.0L;
    long double m2 = 0.0L;
};

inline Moments1 single_moment(double value) {
    return Moments1{1.0L, static_cast<long double>(value), 0.0L};
}

inline Moments1 merge(const Moments1& a, const Moments1& b) {
    if (a.count == 0.0L) return b;
    if (b.count == 0.0L) return a;
    const long double count = a.count + b.count;
    const long double delta = b.mean - a.mean;
    const long double weight = a.count * b.count / count;
    Moments1 result;
    result.count = count;
    result.mean = a.mean + delta * b.count / count;
    result.m2 = a.m2 + b.m2 + delta * delta * weight;
    return result;
}

class SlidingMoments1 {
public:
    void push(double value) {
        Entry entry{value, single_moment(value)};
        if (!back_.empty()) entry.aggregate = merge(back_.back().aggregate,
                                                    entry.aggregate);
        back_.push_back(entry);
        ++size_;
    }

    void pop() {
        if (front_.empty()) transfer();
        front_.pop_back();
        --size_;
    }

    std::size_t size() const { return size_; }

    Moments1 moments() const {
        if (front_.empty()) {
            return back_.empty() ? Moments1{} : back_.back().aggregate;
        }
        if (back_.empty()) return front_.back().aggregate;
        return merge(front_.back().aggregate, back_.back().aggregate);
    }

private:
    struct Entry {
        double value;
        Moments1 aggregate;
    };

    void transfer() {
        while (!back_.empty()) {
            Entry entry = back_.back();
            back_.pop_back();
            entry.aggregate = single_moment(entry.value);
            if (!front_.empty()) {
                entry.aggregate = merge(entry.aggregate,
                                        front_.back().aggregate);
            }
            front_.push_back(entry);
        }
    }

    std::vector<Entry> back_;
    std::vector<Entry> front_;
    std::size_t size_ = 0;
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
    const long double count = a.count + b.count;
    const long double dx = b.mean_x - a.mean_x;
    const long double dy = b.mean_y - a.mean_y;
    const long double weight = a.count * b.count / count;
    Moments2 result;
    result.count = count;
    result.mean_x = a.mean_x + dx * b.count / count;
    result.mean_y = a.mean_y + dy * b.count / count;
    result.m2_x = a.m2_x + b.m2_x + dx * dx * weight;
    result.m2_y = a.m2_y + b.m2_y + dy * dy * weight;
    result.co_moment = a.co_moment + b.co_moment + dx * dy * weight;
    return result;
}

class SlidingMoments2 {
public:
    void push(double x, double y) {
        Entry entry{x, y, single_moment2(x, y)};
        if (!back_.empty()) entry.aggregate = merge(back_.back().aggregate,
                                                    entry.aggregate);
        back_.push_back(entry);
        ++size_;
    }

    void pop() {
        if (front_.empty()) transfer();
        front_.pop_back();
        --size_;
    }

    std::size_t size() const { return size_; }

    Moments2 moments() const {
        if (front_.empty()) {
            return back_.empty() ? Moments2{} : back_.back().aggregate;
        }
        if (back_.empty()) return front_.back().aggregate;
        return merge(front_.back().aggregate, back_.back().aggregate);
    }

private:
    struct Entry {
        double x;
        double y;
        Moments2 aggregate;
    };

    void transfer() {
        while (!back_.empty()) {
            Entry entry = back_.back();
            back_.pop_back();
            entry.aggregate = single_moment2(entry.x, entry.y);
            if (!front_.empty()) {
                entry.aggregate = merge(entry.aggregate,
                                        front_.back().aggregate);
            }
            front_.push_back(entry);
        }
    }

    std::vector<Entry> back_;
    std::vector<Entry> front_;
    std::size_t size_ = 0;
};

}  // namespace rflux::detail
