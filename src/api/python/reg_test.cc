
#include <iostream>
#include "src/api/python/distdl_api_registry.hh"

namespace py = pybind11;

namespace distdl {

DISTDL_API_PYBIND11_MODULE("test", m) {
    m.def("reg_test", []() { std::cout << "this is reg_test" << std::endl; });
}

} // namespace distdl
