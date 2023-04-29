include (CheckCXXSourceCompiles)
include (CMakePushCheckState)

cmake_push_check_state ()

file(
    GLOB_RECURSE
    distdl_all_src
    "${PROJECT_SOURCE_DIR}/src/core/*.*"
    "${PROJECT_SOURCE_DIR}/src/api/*.*"
)

# set BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS ON)

add_library(distdl
    ${distdl_all_src}
)

target_include_directories(distdl PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(distdl
    PRIVATE
        ch_contrib::pybind11
        python3.8
)

# unset BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS OFF)
