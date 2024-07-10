#include "onnx_ir/ir.h"

#include <gtest/gtest.h>

namespace my_ai_training::ir {
namespace {

TEST(IrTest, Graph) { std::cout << "IrTest::Graph\n"; }

// 用于跟踪析构函数调用的简单类
class TestDestructor {
 public:
  static int destructor_calls;

  ~TestDestructor() { ++destructor_calls; }

  static void ResetCalls() { destructor_calls = 0; }
};

int TestDestructor::destructor_calls = 0;

TEST(IrTest, ResourceGuard) {
  TestDestructor::ResetCalls();
  {
    auto destructor = []() { TestDestructor(); };
    auto real_destructor = []() { TestDestructor td; };
    ResourceGuard guard(real_destructor);
  }
  EXPECT_EQ(1, TestDestructor::destructor_calls);
}

TEST(IrTest, ONNX_ASSERT) { ONNX_ASSERT(1 < 2); }

}  // namespace
}  // namespace my_ai_training::ir