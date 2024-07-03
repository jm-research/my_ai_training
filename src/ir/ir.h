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
  bool hasUses() const {
    for (auto o : outputs()) {
      if (!o->uses().empty()) return true;
    }
    return false;
  }
  void replaceAllUsesWith(Node* n) {
    ONNX_ASSERT(outputs().size() == n->outputs().size());
    size_t nOutputs = outputs().size();
    for (size_t i = 0; i < nOutputs; i++) {
      outputs()[i]->replaceAllUsesWith(n->outputs()[i]);
    }
  }
  // lots of things like chunk have a single input or single output, so we have
  // a helper to make accessing it easier
  Value* input() {
    ONNX_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  Value* output() {
    ONNX_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const Value* input() const {
    ONNX_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  Value* output() const {
    ONNX_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  // Access a particular input.  This is a checked index.
  Value* input(size_t i) { return inputs_.at(i); }
  const Value* input(size_t i) const { return inputs_.at(i); }

  // Graphs

  // Note [Topological invariant]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // We always maintain an up-to-date topological ordering of all nodes via
  // the next()/prev() links.  All transformations to graphs must preserve
  // this topological ordering: for example, it is only valid to 'addInput'
  // with an input which is topologically before the current node.
  //
  // Usually, it is obvious whether or not topological order is maintained;
  // for example, if you are adding nodes to the end of the topsort, it's
  // impossible for them to refer to inputs that are not in the topsort.
  // If it is not obvious, please comment accordingly.

  // Add 'node' as an input to 'this' at the end of existing
  // arguments.  Returns the added node for ease of chaining.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.addInput(%4)
  // Result:  %3 = f(%1, %2, %4)
  Value* addInput(Value* node) {
    ONNX_ASSERT(graph_ == node->owningGraph());
    node->uses_in_current_graph_.emplace_back(this, inputs_.size());
    inputs_.push_back(node);
    return node;
  }

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  Value* replaceInput(size_t i, Value* newValue) {
    ONNX_ASSERT(newValue->owningGraph() == graph_);
    Value* old = dropInput(i);
    inputs_[i] = newValue;
    newValue->uses_in_current_graph_.emplace_back(this, i);
    return old;
  }

  // Replace all occurrences of 'from' in the inputs of this
  // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
  //
  // Given:   %3 = f(%1, %2, %1)
  // Execute: %3.replaceInputWith(%1, %4)
  // Result:  %3 = f(%4, %2, %4)
  void replaceInputWith(Value* from, Value* to) {
    ONNX_ASSERT(from->owningGraph() == graph_);
    ONNX_ASSERT(to->owningGraph() == graph_);
    size_t i = 0;
    for (auto input : inputs()) {
      if (input == from) replaceInput(i, to);
      i++;
    }
  }

  Value* addOutput() {
    outputs_.push_back(new Value(this, outputs_.size()));
    return outputs_.back();
  }

  void eraseOutput(size_t i);

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertBefore(%4)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  Node* insertBefore(Node* n) {
    ONNX_ASSERT(n->inGraphList());
    insertAfter(n->prev());
    return this;
  }

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given: %3 = f(%1, %2)
  //        %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertAfter(%4)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%1)
  Node* insertAfter(Node* n) {
    ONNX_ASSERT(!inGraphList() && n->inGraphList());
    Node* next = n->next();
    n->next() = this;
    this->prev() = n;
    this->next() = next;
    next->prev() = this;
    return this;
  }

  // Move 'this' (already in the graph) after 'n' in the topological order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.moveAfter(%3)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  //
  void moveAfter(Node* n) {
    removeFromList();
    insertAfter(n);
  }

  // Move a node 'n' (already in the graph) before 'this' in the topological
  // order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %3.moveBefore(%2)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  void moveBefore(Node* n) {
    removeFromList();
    insertBefore(n);
  }

  // Remove the input at 'i' from this node.
  //
  // WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
  // removeInput.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeInput(1)
  // Result: %3 = f(%1)
  void removeInput(size_t i) {
    dropInput(i);
    // everything after this input shifts left,
    // so we need to update their use offsets to match
    for (size_t j = i + 1; j < inputs_.size(); j++) {
      auto it = findUseForInput(j);
      it->offset--;
    }
    inputs_.erase(inputs_.begin() + i);
  }

  // Remove all inputs from a node.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeAllInputs()
  // Result: %3 = f()
  void removeAllInputs() {
    for (size_t i = 0; i < inputs().size(); ++i) dropInput(i);
    inputs_.clear();
  }

  // Check whether this node is before node n in the graph.
  bool isBefore(Node* n);

  // iterators of the node list starting at this node
  // useful for resuming a search starting at this node
  graph_node_list_iterator iterator();
  graph_node_list_iterator reverseIterator();
  const_graph_node_list_iterator iterator() const;
  const_graph_node_list_iterator reverseIterator() const;

  // Remove 'this' from the instruction list and deallocate it.
  //
  // Invariant: no outputs of 'this' may have any uses.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.destroy()
  // Result: %3 = g(%1)
  void destroy();

  // Dynamically cast this node to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid..
  //
  // Example usage: if(auto s = n.cast<Select>()) { ... }
  //
  // TODO: Make this const correct
  template <typename T>
  T* cast() {
    if (T::Kind == kind()) return static_cast<T*>(this);
    return nullptr;
  }
  template <typename T>
  T* expect() {
    ONNX_ASSERTM(T::Kind == kind(), "expected a %s but found a %s",
                 T::Kind.toString(), kind().toString());
    return static_cast<T*>(this);
  }

  virtual ~Node() = default;

 private:
  use_list::iterator findUseForInput(size_t i) {
    auto& input_uses = inputs_[i]->uses_in_current_graph_;
    auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
    ONNX_ASSERT(use_it != input_uses.end());
    return use_it;
  }

  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  Value* dropInput(size_t i) {
    ONNX_ASSERT(i < inputs_.size());
    auto input_node = inputs_[i];
    auto use_it = findUseForInput(i);
    input_node->uses_in_current_graph_.erase(use_it);
    inputs_[i] = nullptr;
    return input_node;
  }

  // 如果一个节点有下一个节点 (next() 不是 nullptr)，那么它肯定在链表中；
  // 如果一个节点没有前一个节点 (prev() 是
  // nullptr)，它可能是链表的头节点，也意味着它在链表中。
  bool inGraphList() const {
    ONNX_ASSERT(next() != nullptr || prev() == nullptr);
    return next() != nullptr;
  }
  void removeFromList() {
    ONNX_ASSERT(inGraphList());
    Node* next = this->next();
    Node* prev = this->prev();
    prev->next() = next;
    next->prev() = prev;
    this->next() = nullptr;
    this->prev() = nullptr;
  }

 protected:
  // subclasses must override
  // this function is used by createClone to initialize a new version
  // of a node in another graph. It should allocate a new instance of the same
  // concrete type as 'this', but in graph 'g' which might be different
  // than graph_
  virtual Node* allocNewInstance(Graph* g) { return new Node(g, kind()); }
};

struct Graph final {
  MY_AI_TRAINING_DISALLOW_COPY_AND_ASSIGN(Graph);
  friend struct Node;
  friend struct Value;

 private:
  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_set<const Node*> all_nodes;
  std::unordered_set<const Value*> all_values;
  size_t next_unique_;

  size_t new_node_stage_;

  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Node* const output_;
  Node* const input_;
  // Create an independent node list for those initializers do not exist in
  // input
  Node* const initializer_node_;

  std::vector<std::string> initializer_names_;

  bool has_name_;
  std::string name_;
  bool has_doc_string_;
  std::string doc_string_;
};

}  // namespace my_ai_training::ir
