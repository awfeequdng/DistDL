#pragma once

#define NO_IMPORT_ARRAY
#include "src/extension/python/numpy_internal.hh"

namespace distdl {

class NumPyArrayPtr final {
 public:
  NumPyArrayPtr(PyObject* obj)
      : internal_(std::make_shared<numpy::NumPyArrayInternal>(obj, []() -> void {})) {}
  NumPyArrayPtr(PyObject* obj, const std::function<void()>& deleter)
      : internal_(std::make_shared<numpy::NumPyArrayInternal>(obj, deleter)) {}

  void* data() const { return internal_->data(); }

  size_t size() const { return internal_->size(); }

 private:
  std::shared_ptr<numpy::NumPyArrayInternal> internal_;
};

}  // namespace distdl

