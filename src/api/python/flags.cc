
#include "src/api/python/distdl_api_registry.hh"

namespace distdl {

DISTDL_API_PYBIND11_MODULE("flags", m) {

    m.def("use_cxx11_abi", []{
    #if _GLIBCXX_USE_CXX11_ABI == 1
        return true;
    #else
        return false;
    #endif // _GLIBCXX_USE_CXX11_ABI
    });

#define STRINGIFY(x) STRINGIFY_(x)
#define STRINGIFY_(x) #x
    m.def("cmake_build_type", []() {
#ifdef DISTDL_CMAKE_BUILD_TYPE
        return std::string(STRINGIFY(DISTDL_CMAKE_BUILD_TYPE));
#else
        return std::string("Undefined");
#endif  // DISTDL_CMAKE_BUILD_TYPE
    });
#undef STRINGIFY
#undef STRINGIFY_
}

} // namespace distdl
