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

#include "ir/array_ref.h"
#include "ir/assertions.h"
#include "ir/graph_node_list.h"
#include "ir/interned_strings.h"

#define MY_AI_TRAINING_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;                     \
  TypeName& operator=(const TypeName&) = delete

namespace my_ai_training::ir {

namespace {  // internal/private API

std::string toVarName(size_t i) {
  std::ostringstream oss;
  oss << "_v_" << i;
  return oss.str();
}

}  // namespace

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

// Each use is represented by this type, see Node::uses()
// 'user' is the consumer of the value, offset is the index into
// 'user's input this where the produces will be found.
struct Use final {
  Use(Node* user, size_t offset) : user(user), offset(offset) {}
  Node* user;
  size_t offset;
};

static inline bool operator==(const Use& a, const Use& b) {
  return a.user == b.user && a.offset == b.offset;
}

// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
using node_list = std::vector<Node*>;
using value_list = std::vector<Value*>;
using use_list = std::vector<Use>;
using NodeKind = Symbol;

struct Value final {
  MY_AI_TRAINING_DISALLOW_COPY_AND_ASSIGN(Value);
  Value(Node* node, size_t offset);
  Value(Value&&) = default;
  Value& operator=(Value&&) = default;
  ~Value() = default;

 private:
  friend struct Node;
  friend struct Graph;
  Node* node_;
  size_t offset_;
  size_t unique_ = 0;  // unique id
  size_t stage_ = 0;   // 0-forward, 1-backward, 2-double-backward,...
  use_list uses_in_current_graph_;
  bool has_unique_name_;
  std::string unique_name_;
  int32_t elem_type_;
  bool has_sizes_;
  std::vector<Dimension> sizes_;

 public:
  Value* setElemType(int32_t elem_type) {
    elem_type_ = elem_type;
    return this;
  }

  int32_t elemType() const { return elem_type_; }

  bool has_sizes() const { return has_sizes_; }

  Value* setSizes(std::vector<Dimension> sizes) {
    has_sizes_ = true;
    sizes_ = std::move(sizes);
    return this;
  }

  Value* wipeSizes() {
    has_sizes_ = false;
    sizes_ = std::vector<Dimension>();
    return this;
  }

  const std::vector<Dimension>& sizes() const { return sizes_; }

  size_t unique() const { return unique_; }

  bool has_unique_name() const { return has_unique_name_; }

  std::string uniqueName() const {
    if (has_unique_name()) return unique_name_;
    return toVarName(unique());
  }

  Value* setUniqueName(const std::string& name,
                       bool rename_subgraph_captured_nodes = true);
  Value* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  size_t stage() const { return stage_; }
  Node* node() { return node_; }
  size_t offset() const { return offset_; }
  const Node* node() const { return node_; }
  Graph* owningGraph();
  const Graph* owningGraph() const;
  // TODO: make this more const correct
  const use_list uses() const;

  // Replaces all uses of this node with 'newValue'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  void replaceAllUsesWith(Value* newValue);

  Value* copyMetadata(Value* from) {
    setElemType(from->elemType());
    setSizes(from->sizes());
    if (from->has_unique_name()) {
      setUniqueName(from->uniqueName());
    }
    return this;
  }
};

struct Node {
  MY_AI_TRAINING_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
  friend struct Value;
  friend graph_node_list;
  friend const_graph_node_list;
  friend graph_node_list_iterator;
  friend const_graph_node_list_iterator;

 private:
  // each node but Return/Param
  // is associated with exactly one place in the node list...
  // of the graph_
  // this circular is a doubly-linked list, the Return node is used as the
  // sentinel for the beginning and end of the list such that the list never has
  // null pointers next_in_graph[0] is next pointer next_in_graph[1] is prev
  // pointer using an array to allow the same iterator class for forward and
  // reverse node lists This list represents a topological sort

  Node* next_in_graph[2] = {nullptr, nullptr};
  Node*& next() { return next_in_graph[kNextDirection]; }
  Node*& prev() { return next_in_graph[kPrevDirection]; }
  Node* const& next() const { return next_in_graph[kNextDirection]; }
  Node* const& prev() const { return next_in_graph[kPrevDirection]; }

  const NodeKind kind_;
  std::vector<Value*> inputs_;
  std::vector<Value*> outputs_;
  Graph* graph_;
  size_t stage_;
  bool has_name_;
  std::string name_;
  bool has_domain_;
  std::string domain_;
  bool has_doc_string_;
  std::string doc_string_;
  bool has_overload_;
  std::string overload_;

 protected:
  Node(Graph* graph_, NodeKind kind_);  // defined after graph

 public:
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }
  bool has_domain() const { return has_domain_; }
  const std::string& domain() const { return domain_; }
  void setDomain(std::string domain) {
    has_domain_ = true;
    domain_ = std::move(domain);
  }
  bool has_overload() const { return has_overload_; }
  const std::string& overload() const { return overload_; }
  void setOverload(std::string overload) {
    has_overload_ = true;
    overload_ = std::move(overload);
  }
  bool has_doc_string() const { return has_doc_string_; }
  const std::string& docString() const { return doc_string_; }
  void setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
  }
  NodeKind kind() const { return kind_; }
  Graph* owningGraph() { return graph_; }
  const Graph* owningGraph() const { return graph_; }
  size_t stage() const { return stage_; }
  Node* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  ArrayRef<Value*> inputs() { return inputs_; }
  ArrayRef<const Value*> inputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {inputs_.data(), inputs_.size()};
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  ArrayRef<Value*> outputs() { return outputs_; }
  ArrayRef<const Value*> outputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {outputs_.data(), outputs_.size()};
  }
};

}  // namespace my_ai_training::ir
