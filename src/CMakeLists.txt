

file(
    GLOB_RECURSE
    distdl_all_src
    "${PROJECT_SOURCE_DIR}/src/core/*.*"
    "${PROJECT_SOURCE_DIR}/src/api/*.*"
)

set(DISTDL_PYTHON_DIR "${PROJECT_BINARY_DIR}/python")

# set BUILD_SHARED_LIBS
# set(BUILD_SHARED_LIBS ON)

pybind11_add_module(distdl ${distdl_all_src})

set_target_properties(distdl PROPERTIES PREFIX "_")

set_target_properties(distdl PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                        "${DISTDL_PYTHON_DIR}/distdl")

# set_property(TARGET distdl PROPERTY CXX_VISIBILITY_PRESET "default")


# unset BUILD_SHARED_LIBS
# set(BUILD_SHARED_LIBS OFF)
