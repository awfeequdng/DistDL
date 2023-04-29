#pragma once

#include <pybind11/pybind11.h>
#include <map>
#include <vector>
#include <functional>

#include "src/core/common/preprocessor.hh"

namespace distdl {

class DistDLModuleRegistry {
 public:
  DistDLModuleRegistry() = delete;
  ~DistDLModuleRegistry() = delete;

  static void Register(std::string module_path, std::function<void(pybind11::module&)> BuildModule);
  static void ImportAll(pybind11::module& m);

 private:
  static void BuildSubModule(const std::string& module_path, pybind11::module& m,
                      const std::function<void(pybind11::module&)>& BuildModule);
};

}  // namespace distdl

#define DISTDL_API_PYBIND11_MODULE(module_path, m)                                              \
  static void DISTDL_PP_CAT(DistDLApiPythonModule, __LINE__)(pybind11::module&);                    \
  namespace {                                                                                    \
  struct DistDLApiRegistryInit {                                                                     \
    DistDLApiRegistryInit() {                                                                        \
      ::distdl::DistDLModuleRegistry::Register(module_path,                                   \
                                                  &DISTDL_PP_CAT(DistDLApiPythonModule, __LINE__)); \
    }                                                                                            \
  };                                                                                             \
  DistDLApiRegistryInit of_api_registry_init;                                                        \
  }                                                                                              \
  static void DISTDL_PP_CAT(DistDLApiPythonModule, __LINE__)(pybind11::module & m)

