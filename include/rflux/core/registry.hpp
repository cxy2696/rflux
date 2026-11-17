#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rflux::core {

/*
 * Thread-safe, type-erased algorithm registry.
 *
 * Creators are copied under the registry lock and invoked outside it, so a
 * factory can be used by concurrent workers without holding a global lock while
 * an algorithm is constructed. Duplicate names are rejected unless replace is
 * explicitly requested.
 */
template <typename Interface, typename... Args>
class AlgorithmRegistry {
public:
    using Creator = std::function<std::unique_ptr<Interface>(Args...)>;

    static AlgorithmRegistry& instance() {
        static AlgorithmRegistry registry;
        return registry;
    }

    void add(std::string name, Creator creator, bool replace = false) {
        if (name.empty()) {
            throw std::invalid_argument("algorithm name must not be empty");
        }
        if (!creator) {
            throw std::invalid_argument("algorithm creator must be callable");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!replace && creators_.find(name) != creators_.end()) {
            throw std::invalid_argument("algorithm already registered: " + name);
        }
        creators_[std::move(name)] = std::move(creator);
    }

    [[nodiscard]] std::unique_ptr<Interface> create(const std::string& name,
                                                     Args... args) const {
        Creator creator;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = creators_.find(name);
            if (found == creators_.end()) {
                throw std::invalid_argument("unknown algorithm: " + name);
            }
            creator = found->second;
        }
        return creator(std::forward<Args>(args)...);
    }

    [[nodiscard]] bool contains(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return creators_.find(name) != creators_.end();
    }

    [[nodiscard]] std::vector<std::string> names() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> result;
        result.reserve(creators_.size());
        for (const auto& entry : creators_) result.push_back(entry.first);
        std::sort(result.begin(), result.end());
        return result;
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return creators_.size();
    }

private:
    AlgorithmRegistry() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Creator> creators_;
};

}  // namespace rflux::core
