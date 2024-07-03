#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#define MY_AI_TRAINING_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;                     \
  TypeName& operator=(const TypeName&) = delete

namespace my_ai_training::ir {

struct Graph;

struct Node;

struct Value;

class ResourceGuard final {
  std::function<void()> destructor_;
  bool released_;

 public:
  MY_AI_TRAINING_DISALLOW_COPY_AND_ASSIGN(ResourceGuard);
  explicit ResourceGuard(std::function<void()> destructor)
      : destructor_(std::move(destructor)), released_(false) {}
  ResourceGuard(ResourceGuard&& other) = default;
  ResourceGuard& operator=(ResourceGuard&& other) = default;

  ~ResourceGuard() {
    if (!released_) destructor_();
  }

  void release() { released_ = true; }
};

}  // namespace my_ai_training::ir
