// ************************
//
// NOTE: Do NOT include this file (numpy_internal.h) directly.
// Include numpy.h instead.
//
// ************************


// PyArrayObject cannot be forward declared, or a compile error will occur

// https://numpy.org/doc/stable/reference/c-api/array.html?highlight=array%20api#importing-the-api
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL oneflow_ARRAY_API
#include <numpy/arrayobject.h>

#include <functional>

namespace distdl {

namespace numpy {

class NumPyArrayInternal final {
 public:
  NumPyArrayInternal(PyObject* obj, const std::function<void()>& deleter);
  ~NumPyArrayInternal();

  void* data() const { return data_; }

  size_t size() const { return size_; }

 private:
  PyArrayObject* obj_;
  void* data_;
  size_t size_;
  std::function<void()> deleter_;
};

void InitNumpyCAPI();

}  // namespace numpy
}  // namespace distdl
