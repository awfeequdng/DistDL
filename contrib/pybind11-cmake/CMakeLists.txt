set(PYBIND11_SOURCE_DIR "${DistDL_SOURCE_DIR}/contrib/pybind11")

add_library(_pybind11 INTERFACE)
target_include_directories(_pybind11 INTERFACE ${PYBIND11_SOURCE_DIR}/include)
add_library(ch_contrib::pybind11 ALIAS _pybind11)

MESSAGE(STATUS "pybind11 header dir: ${PYBIND11_SOURCE_DIR}/include")
