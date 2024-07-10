#pragma once

#include "ncnn/allocator.h"

namespace ncnn {

class Mat {
 public:
  // empty
  Mat();
  // vec
  Mat(int w, size_t elemsize = 4u, Allocator* allocator = 0);
  // image
  Mat(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
  // dim
  Mat(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
  // cube
  Mat(int w, int h, int d, int c, size_t elemsize = 4u,
      Allocator* allocator = 0);
  // packed vec
  Mat(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
  // packed image
  Mat(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
  // packed dim
  Mat(int w, int h, int c, size_t elemsize, int elempack,
      Allocator* allocator = 0);
  // packed cube
  Mat(int w, int h, int d, int c, size_t elemsize, int elempack,
      Allocator* allocator = 0);
  // copy
  Mat(const Mat& m);
  // external vec
  Mat(int w, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
  // external image
  Mat(int w, int h, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
  // external dim
  Mat(int w, int h, int c, void* data, size_t elemsize = 4u,
      Allocator* allocator = 0);
  // external cube
  Mat(int w, int h, int d, int c, void* data, size_t elemsize = 4u,
      Allocator* allocator = 0);
  // external packed vec
  Mat(int w, void* data, size_t elemsize, int elempack,
      Allocator* allocator = 0);
  // external packed image
  Mat(int w, int h, void* data, size_t elemsize, int elempack,
      Allocator* allocator = 0);
  // external packed dim
  Mat(int w, int h, int c, void* data, size_t elemsize, int elempack,
      Allocator* allocator = 0);
  // external packed cube
  Mat(int w, int h, int d, int c, void* data, size_t elemsize, int elempack,
      Allocator* allocator = 0);
  // release
  ~Mat();
  // assign
  Mat& operator=(const Mat& m);
  // set all
  void fill(float v);
  void fill(int v);
};

}  // namespace ncnn