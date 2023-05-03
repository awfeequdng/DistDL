/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <pybind11/pybind11.h>
#include "src/extension/python/numpy_internal.hh"
#include <glog/logging.h>

namespace py = pybind11;

namespace distdl {

namespace numpy {

NumPyArrayInternal::NumPyArrayInternal(PyObject* obj, const std::function<void()>& deleter)
    : obj_((PyArrayObject*)obj), deleter_(deleter) {
    if (!PyArray_Check(obj)) {
        LOG(ERROR) << "The object is not a numpy array.";
        exit(0);
    }
    if (!PyArray_ISCONTIGUOUS(obj_)) {
        LOG(ERROR) << "Contiguous array is expected.";
        exit(0);
    }
    size_ = PyArray_SIZE(obj_);
    data_ = PyArray_DATA(obj_);
}

NumPyArrayInternal::~NumPyArrayInternal() {
  if (deleter_) { deleter_(); }
}

// Executing any numpy c api before _import_array() results in segfault
// NOTE: this InitNumpyCAPI() works because of `PY_ARRAY_UNIQUE_SYMBOL`
// defined in numpy_internal.h
// Reference:
// https://numpy.org/doc/stable/reference/c-api/array.html#importing-the-api
void InitNumpyCAPI() {
    LOG(INFO) << "InitNumpyCAPI()";
    // VLOG(2) << "InitNumpyCAPI()" << std::endl;

    if (PyArray_API != nullptr) {
      return;
    }
    // CHECK_EQ_OR_RETURN(_import_array(), 0)
    if (_import_array() != 0) {
        LOG(INFO)<< ". Unable to import Numpy array, try to upgrade Numpy version!";
    }
}

}  // namespace numpy
}  // namespace oneflow
