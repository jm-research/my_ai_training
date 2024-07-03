#pragma once

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

#include "assertions.h"

#define MY_AI_TRAINING_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;                     \
  TypeName& operator=(const TypeName&) = delete

namespace my_ai_training::ir {

struct Graph;

struct Node;

struct Value;

// resource guard helper class, raii helper.
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

struct Dimension final {
  Dimension() : is_unknown(true), is_int(false), dim(-1) {}
  Dimension(std::string param)  // NOLINT
      : is_unknown(false), is_int(false), dim(-1), param(std::move(param)) {}
  Dimension(int64_t dim)  // NOLINT
      : is_unknown(false), is_int(true), dim(dim) {}

  bool is_unknown;    // 表示维度是否是未知的
  bool is_int;        // 表示已知的维度是否是一个整数
  int64_t dim;        // 存储已知的整数维度值
  std::string param;  // 存储非整数形式的维度信息
};

enum class AttributeKind : uint8_t {
  // float, float list, int, int list, string, string list,
  // tensor, tensor list, subgraph, subgraph list. type proto, type proto list
  f,
  fs,
  i,
  is,
  s,
  ss,
  t,
  ts,
  g,
  gs,
  tp,
  tps
};

static inline const char* toString(AttributeKind kind) {
  static constexpr const char* names[] = {"f", "fs", "i", "is", "s",  "ss",
                                          "t", "ts", "g", "gs", "tp", "tps"};
  ONNX_ASSERT(size_t(kind) < sizeof(names) / sizeof(const char*));
  return names[int(kind)];
}

}  // namespace my_ai_training::ir
