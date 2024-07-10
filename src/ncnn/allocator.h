#pragma once

#include <stdlib.h>

#include "ncnn/platform.h"

namespace ncnn {

// the alignment of all the allocated buffers
#if NCNN_AVX512
#define NCNN_MALLOC_ALIGN 64
#elif NCNN_AVX
#define NCNN_MALLOC_ALIGN 32
#else
#define NCNN_MALLOC_ALIGN 16
#endif

// 设置了一个常数值64。这个值是为了在内存分配时预留额外的字节，
// 以防某些优化的内核在循环中稍微越界读取数据，导致段错误。
// 这种“过读”通常发生在边界处，通过预留额外的内存可以避免此类错误。
#define NCNN_MALLOC_OVERREAD 64

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template <typename _Tp>
static NCNN_FORCEINLINE _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp)) {
  return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is
// divisible by n sz Buffer size to align n Alignment size that must be a power
// of two
static NCNN_FORCEINLINE size_t alignSize(size_t sz, int n) {
  return (sz + n - 1) & -n;
}

static NCNN_FORCEINLINE void* fastMalloc(size_t size) {
#if _MSC_VER
  return _aligned_malloc(size, NCNN_MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && \
        _POSIX_C_SOURCE >= 200112L ||              \
    (__ANDROID__ && __ANDROID_API__ >= 17)
  void* ptr = 0;
  if (posix_memalign(&ptr, NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD))
    ptr = 0;
  return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
  return memalign(NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD);
#else
  unsigned char* udata = (unsigned char*)malloc(
      size + sizeof(void*) + NCNN_MALLOC_ALIGN + NCNN_MALLOC_OVERREAD);
  if (!udata) return 0;
  unsigned char** adata =
      alignPtr((unsigned char**)udata + 1, NCNN_MALLOC_ALIGN);
  adata[-1] = udata;
  return adata;
#endif
}

static NCNN_FORCEINLINE void fastFree(void* ptr) {
  if (ptr) {
#if _MSC_VER
    _aligned_free(ptr);
#elif (defined(__unix__) || defined(__APPLE__)) && \
        _POSIX_C_SOURCE >= 200112L ||              \
    (__ANDROID__ && __ANDROID_API__ >= 17)
    free(ptr);
#elif __ANDROID__ && __ANDROID_API__ < 17
    free(ptr);
#else
    unsigned char* udata = ((unsigned char**)ptr)[-1];
    free(udata);
#endif
  }
}

class NCNN_EXPORT Allocator {
 public:
  virtual ~Allocator();
  virtual void* fastMalloc(size_t size) = 0;
  virtual void fastFree(void* ptr) = 0;
};

class PoolAllocatorPrivate;
class NCNN_EXPORT PoolAllocator : public Allocator {
 public:
  PoolAllocator();
  ~PoolAllocator();

  // ratio range 0 ~ 1
  // default cr = 0
  void set_size_compare_ratio(float scr);

  // budget drop threshold
  // default threshold = 10
  void set_size_drop_threshold(size_t);

  // release all budgets immediately
  void clear();

  virtual void* fastMalloc(size_t size);
  virtual void fastFree(void* ptr);

 private:
  PoolAllocator(const PoolAllocator&);
  PoolAllocator& operator=(const PoolAllocator&);

 private:
  PoolAllocatorPrivate* const d;
};

class UnlockedPoolAllocatorPrivate;
class NCNN_EXPORT UnlockedPoolAllocator : public Allocator {
 public:
  UnlockedPoolAllocator();
  ~UnlockedPoolAllocator();

  // ratio range 0 ~ 1
  // default cr = 0
  void set_size_compare_ratio(float scr);

  // budget drop threshold
  // default threshold = 10
  void set_size_drop_threshold(size_t);

  // release all budgets immediately
  void clear();

  virtual void* fastMalloc(size_t size);
  virtual void fastFree(void* ptr);

 private:
  UnlockedPoolAllocator(const UnlockedPoolAllocator&);
  UnlockedPoolAllocator& operator=(const UnlockedPoolAllocator&);

 private:
  UnlockedPoolAllocatorPrivate* const d;
};

}  // namespace ncnn