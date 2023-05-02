include (cmake/python.cmake)
include (cmake/pybind11.cmake)

function(distdl_add_executable)
  add_executable(${ARGV})
  set_compile_options_to_distdl_target(${ARGV0})
endfunction()

function(distdl_add_library)
  add_library(${ARGV})
  set_compile_options_to_distdl_target(${ARGV0})
endfunction()

# source_group
set(distdl_platform "linux")

file(
  GLOB_RECURSE
  distdl_all_src
  "${PROJECT_SOURCE_DIR}/src/core/*.*"
  "${PROJECT_SOURCE_DIR}/src/user/*.*"
  "${PROJECT_SOURCE_DIR}/src/extension/*.*"
  "${PROJECT_SOURCE_DIR}/src/api/*.*")

foreach(distdl_single_file ${distdl_all_src})
  # Verify whether this file is for other platforms

  if("${distdl_single_file}" MATCHES
     "^${PROJECT_SOURCE_DIR}/src/(core|user)/.*\\.(h|hh)$")
    list(APPEND distdl_all_obj_cc ${distdl_single_file})
    set(group_this ON)
  endif()

  if("${distdl_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/src/(core|user)/.*\\.(cuh|cu)$")
    if(BUILD_CUDA)
      list(APPEND distdl_all_obj_cc ${distdl_single_file})
    endif()
    set(group_this ON)
  endif()

  if("${distdl_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/src/(core|user)/.*\\.proto$")
    list(APPEND distdl_all_proto ${distdl_single_file})
    #list(APPEND distdl_all_obj_cc ${distdl_single_file})   # include the proto file in the project
    set(group_this ON)
  endif()

  if(BUILD_PYTHON)

    if("${distdl_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/src/api/python/.*\\.(hh|cc)$")
      list(APPEND distdl_pybind_obj_cc ${distdl_single_file})
      set(group_this ON)
    endif()

    if("${distdl_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/src/extension/.*\\.(c|h|hh|cc)$")
      list(APPEND distdl_pyext_obj_cc ${distdl_single_file})
      set(group_this ON)
    endif()
  endif(BUILD_PYTHON)

  if("${distdl_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/src/(core|user)/.*\\.cc$")
    if("${distdl_single_file}" MATCHES
       "^${PROJECT_SOURCE_DIR}/src/(core|user)/.*_test\\.cc$")
      # test file
      list(APPEND distdl_all_test_cc ${distdl_single_file})
      # skip if GRPC not enabled
    else()
      list(APPEND distdl_all_obj_cc ${distdl_single_file})
    endif()
    set(group_this ON)
  endif()
  if(group_this)
    file(RELATIVE_PATH distdl_relative_file ${PROJECT_SOURCE_DIR}/distdl/core/
         ${distdl_single_file})
    get_filename_component(distdl_relative_path ${distdl_relative_file} PATH)
    string(REPLACE "/" "\\" group_name ${distdl_relative_path})
    source_group("${group_name}" FILES ${distdl_single_file})
  endif()
endforeach()

# clang format
add_custom_target(
  distdl_format
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_license_format.py -i
          ${CMAKE_CURRENT_SOURCE_DIR}/distdl --fix
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_license_format.py -i
          ${DISTDL_PYTHON_DIR} --fix --exclude="src/include" --exclude="src/core"
  # COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_clang_format.py --source_dir
  #         ${CMAKE_CURRENT_SOURCE_DIR}/src --fix --quiet
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_py_format.py --source_dir
          ${CMAKE_CURRENT_SOURCE_DIR}/python --fix
)

# generate version
set(DISTDL_GIT_VERSION_DIR ${CMAKE_CURRENT_BINARY_DIR}/distdl_git_version)
set(DISTDL_GIT_VERSION_FILE ${DISTDL_GIT_VERSION_DIR}/version.cc)
set(DISTDL_GIT_VERSION_DUMMY_FILE ${DISTDL_GIT_VERSION_DIR}/_version.cc)
add_custom_target(distdl_git_version_create_dir COMMAND ${CMAKE_COMMAND} -E make_directory
                                                    ${DISTDL_GIT_VERSION_DIR})
add_custom_command(
  OUTPUT ${DISTDL_GIT_VERSION_DUMMY_FILE}
  COMMAND ${CMAKE_COMMAND} -DDISTDL_GIT_VERSION_FILE=${DISTDL_GIT_VERSION_FILE}
          -DDISTDL_GIT_VERSION_ROOT=${PROJECT_SOURCE_DIR} -DBUILD_GIT_VERSION=${BUILD_GIT_VERSION} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_version.cmake
  DEPENDS distdl_git_version_create_dir)
add_custom_target(distdl_git_version DEPENDS ${DISTDL_GIT_VERSION_DUMMY_FILE})
set_source_files_properties(${DISTDL_GIT_VERSION_FILE} PROPERTIES GENERATED TRUE)
list(APPEND distdl_all_obj_cc ${DISTDL_GIT_VERSION_FILE})

set(distdl_proto_python_dir "${PROJECT_BINARY_DIR}/distdl_proto_python")

# # proto obj lib
add_custom_target(make_pyproto_dir ALL COMMAND ${CMAKE_COMMAND} -E make_directory
                                               ${distdl_proto_python_dir})
# foreach(proto_name ${distdl_all_proto})
#   file(RELATIVE_PATH proto_rel_name ${PROJECT_SOURCE_DIR} ${proto_name})
#   list(APPEND distdl_all_rel_protos ${proto_rel_name})
# endforeach()

# relative_protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROJECT_SOURCE_DIR} ${distdl_all_rel_protos})

# distdl_add_library(distdl_protoobj SHARED ${PROTO_SRCS} ${PROTO_HDRS})
# add_dependencies(distdl_protoobj make_pyproto_dir protobuf)
# target_link_libraries(distdl_protoobj protobuf_imported)

# include(functional)
# generate_functional_api_and_pybind11_cpp(FUNCTIONAL_GENERATED_SRCS FUNCTIONAL_GENERATED_HRCS
#                                          FUNCTIONAL_PYBIND11_SRCS ${PROJECT_SOURCE_DIR})
# distdl_add_library(of_functional_obj OBJECT ${FUNCTIONAL_GENERATED_SRCS}
#                     ${FUNCTIONAL_GENERATED_HRCS})
# target_link_libraries(distdl_functional_obj LLVMSupportWithHeader glog::glog fmt)
# add_dependencies(distdl_functional_obj prepare_distdl_third_party)

include_directories(${PROJECT_SOURCE_DIR}) # TO FIND: third_party/eigen3/..
include_directories(${PROJECT_BINARY_DIR})

# cc obj lib
distdl_add_library(distdl SHARED ${distdl_all_obj_cc})

# add_dependencies(distdl distdl_protoobj)
# add_dependencies(distdl distdl_functional_obj)
# add_dependencies(distdl distdl_op_schema)
add_dependencies(distdl distdl_git_version)

if(USE_CLANG_FORMAT)
  add_dependencies(distdl distdl_format)
endif()
# if(USE_CLANG_TIDY)
#   add_dependencies(distdl distdl_tidy)
# endif()

# target_compile_definitions(distdl PRIVATE GOOGLE_LOGGING)

set(DISTDL_TOOLS_DIR "${PROJECT_BINARY_DIR}/tools"
    CACHE STRING "dir to put binary for debugging and development")

set(DISTDL_BUILD_ROOT_DIR "${PROJECT_BINARY_DIR}")

# for stack backtrace

set(distdl_libs -ldl -lrt -lunwind)
target_link_libraries(
  distdl
  -Wl,--no-whole-archive
  -Wl,--as-needed
  -ldl
  -lrt)
if(BUILD_CUDA)
  target_link_libraries(distdl CUDA::cudart_static)
endif()
if(WITH_OMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(distdl OpenMP::OpenMP_CXX)
  endif()
endif()

# distdl api common
if(BUILD_PYTHON OR BUILD_CPP_API)
  file(GLOB_RECURSE distdl_api_common_files ${PROJECT_SOURCE_DIR}/src/api/common/*.h
       ${PROJECT_SOURCE_DIR}/src/api/common/*.cc)
  distdl_add_library(distdl_api_common OBJECT ${distdl_api_common_files})
  target_link_libraries(distdl_api_common distdl)
endif()

if(BUILD_PYTHON)

  # py ext lib
  # This library should be static to make sure all python symbols are included in the final ext shared lib,
  # so that it is safe to do wheel audits of multiple pythons version in parallel.
  distdl_add_library(distdl_pyext_obj STATIC ${distdl_pyext_obj_cc})
  target_include_directories(distdl_pyext_obj PRIVATE ${Python_INCLUDE_DIRS}
                                                  ${Python_NumPy_INCLUDE_DIRS})
  target_link_libraries(distdl_pyext_obj distdl ch_contrib::pybind11)
  add_dependencies(distdl_pyext_obj distdl)

  pybind11_add_module(distdl_internal ${distdl_pybind_obj_cc})
  set_compile_options_to_distdl_target(distdl_internal)
  set_property(TARGET distdl_internal PROPERTY CXX_VISIBILITY_PRESET "default")
  set_target_properties(distdl_internal PROPERTIES PREFIX "_")
  set_target_properties(distdl_internal PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                    "${DISTDL_PYTHON_DIR}/distdl")
  target_link_libraries(
    distdl_internal PRIVATE ${distdl_libs} distdl_api_common
                            distdl_pyext_obj)
                            # distdl_pyext_obj glog::glog)
  target_include_directories(distdl_internal PRIVATE ${Python_INCLUDE_DIRS}
                                                      ${Python_NumPy_INCLUDE_DIRS})

  set(gen_pip_args "")
  if(BUILD_CUDA)
    list(APPEND gen_pip_args --cuda=${CUDA_VERSION})
  endif()

  add_custom_target(
    distdl_pyscript_copy ALL
    COMMAND ${CMAKE_COMMAND} -E touch "${distdl_proto_python_dir}/distdl/core/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${distdl_proto_python_dir}/distdl/core"
            "${DISTDL_PYTHON_DIR}/distdl/core"
    COMMAND
      ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_pip_version.py ${gen_pip_args}
      --src=${PROJECT_SOURCE_DIR} --cmake_project_binary_dir=${PROJECT_BINARY_DIR}
      --out=${DISTDL_PYTHON_DIR}/distdl/version.py)

  # source this file to add distdl in PYTHONPATH
  file(WRITE "${PROJECT_BINARY_DIR}/source.sh"
       "export PYTHONPATH=${DISTDL_PYTHON_DIR}:$PYTHONPATH")

  # add_dependencies(distdl_pyscript_copy distdl_protoobj)

endif(BUILD_PYTHON)

if(BUILD_CPP_API)
  file(GLOB_RECURSE distdl_cpp_api_files ${PROJECT_SOURCE_DIR}/src/api/cpp/*.cc
       ${PROJECT_SOURCE_DIR}/src/api/cpp/*.hh)
  list(FILTER distdl_cpp_api_files EXCLUDE REGEX "src/api/cpp/tests")
  distdl_add_library(distdl_cpp SHARED ${distdl_cpp_api_files})
  set_target_properties(distdl_cpp PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${LIBDISTDL_LIBRARY_DIR}"
                                               LIBRARY_OUTPUT_DIRECTORY "${LIBDISTDL_LIBRARY_DIR}")
  target_link_libraries(distdl_cpp PRIVATE ${distdl_libs} distdl_api_common)
endif()

# file(RELATIVE_PATH PROJECT_BINARY_DIR_RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

function(distdl_add_test target_name)
  cmake_parse_arguments(arg "" "TEST_NAME;WORKING_DIRECTORY" "SRCS" ${ARGN})
  distdl_add_executable(${target_name} ${arg_SRCS})
  if(BUILD_CUDA)
    target_link_libraries(${target_name} CUDA::cudart_static)
  endif()
  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${PROJECT_BINARY_DIR}/bin")
  add_test(NAME ${arg_TEST_NAME} COMMAND ${target_name} WORKING_DIRECTORY ${arg_WORKING_DIRECTORY})
  set_tests_properties(
    ${arg_TEST_NAME} PROPERTIES ENVIRONMENT
                                "HTTP_PROXY='';HTTPS_PROXY='';http_proxy='';https_proxy='';")
endfunction()

# build test
if(BUILD_TESTING)
  if(distdl_all_test_cc)
    distdl_add_test(distdl_testexe SRCS ${distdl_all_test_cc} TEST_NAME distdl_test)
    target_link_libraries(distdl_testexe ${distdl_libs})
  endif()

  if(BUILD_CPP_API)
    file(GLOB_RECURSE cpp_api_test_files ${PROJECT_SOURCE_DIR}/src/api/cpp/tests/*.cc)
    distdl_add_test(
      distdl_cpp_api_testexe
      SRCS
      ${cpp_api_test_files}
      TEST_NAME
      distdl_cpp_api_test
      WORKING_DIRECTORY
      ${PROJECT_SOURCE_DIR})
    find_package(Threads REQUIRED)
    target_link_libraries(distdl_cpp_api_testexe distdl_cpp
                          Threads::Threads)
  endif()
endif()

# build include
add_custom_target(distdl_include_copy ALL)

if(BUILD_PYTHON)

  add_dependencies(distdl_include_copy distdl_internal distdl_pyscript_copy)
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/core
    DESTINATION ${DISTDL_INCLUDE_DIR}/distdl
    COMPONENT distdl_py_include
    EXCLUDE_FROM_ALL FILES_MATCHING
    PATTERN *.h
    PATTERN *.hh)
  install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/src
    DESTINATION ${DISTDL_INCLUDE_DIR}
    COMPONENT distdl_py_include
    EXCLUDE_FROM_ALL FILES_MATCHING
    REGEX "src/core/common/.+(h|hh)$"
    # REGEX "src/core/.+(proto)$"
    PATTERN "src/core/common/symbol.hh"
    PATTERN "src/api" EXCLUDE)

  add_custom_target(
    install_distdl_py_include
    COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=distdl_py_include -P
            "${CMAKE_BINARY_DIR}/cmake_install.cmake" DEPENDS distdl_internal)
  add_custom_target(distdl_py ALL)
  add_dependencies(distdl_py distdl_include_copy install_distdl_py_include)

endif(BUILD_PYTHON)

if(BUILD_CPP_API)

  set(LIBDISTDL_DIR ${PROJECT_BINARY_DIR}/libdistdl_cpp)

  install(
    DIRECTORY distdl/api/cpp/
    COMPONENT distdl_cpp_all
    DESTINATION include/distdl
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hh"
    PATTERN "tests" EXCLUDE)
  set(LIBDISTDL_THIRD_PARTY_DIRS)
  checkdirandappendslash(DIR ${PROTOBUF_LIBRARY_DIR} OUTPUT PROTOBUF_LIBRARY_DIR_APPENDED)
  list(APPEND LIBONEFLOW_THIRD_PARTY_DIRS ${PROTOBUF_LIBRARY_DIR_APPENDED})
  if(BUILD_CUDA)
    checkdirandappendslash(DIR ${NCCL_LIBRARY_DIR} OUTPUT NCCL_LIBRARY_DIR_APPENDED)
    list(APPEND LIBONEFLOW_THIRD_PARTY_DIRS ${NCCL_LIBRARY_DIR_APPENDED})
    checkdirandappendslash(DIR ${TRT_FLASH_ATTENTION_LIBRARY_DIR} OUTPUT
                           TRT_FLASH_ATTENTION_LIBRARY_DIR_APPENDED)
    list(APPEND LIBONEFLOW_THIRD_PARTY_DIRS ${TRT_FLASH_ATTENTION_LIBRARY_DIR_APPENDED})
    if(WITH_CUTLASS)
      checkdirandappendslash(DIR ${CUTLASS_LIBRARY_DIR} OUTPUT CUTLASS_LIBRARY_DIR_APPENDED)
      list(APPEND LIBONEFLOW_THIRD_PARTY_DIRS ${CUTLASS_LIBRARY_DIR_APPENDED})
    endif()
  endif()

  # install(
  #   DIRECTORY ${LIBDISTDL_THIRD_PARTY_DIRS}
  #   COMPONENT distdl_cpp_all
  #   DESTINATION lib
  #   FILES_MATCHING
  #   PATTERN "*.so*"
  #   PATTERN "*.a" EXCLUDE
  #   PATTERN "libprotobuf-lite.so*" EXCLUDE
  #   PATTERN "libprotoc.so*" EXCLUDE
  #   PATTERN "cmake" EXCLUDE
  #   PATTERN "pkgconfig" EXCLUDE)

  # install(FILES ${PROJECT_SOURCE_DIR}/cmake/distdl-config.cmake COMPONENT distdl_cpp_all
  #         DESTINATION share)

  set(LIBDISTDL_TARGETS)
  list(
    APPEND
    LIBDISTDL_TARGETS
    distdl_cpp
    distdl
    ${EXTERNAL_TARGETS})

  if(BUILD_TESTING AND BUILD_SHARED_LIBS)
    list(APPEND LIBDISTDL_TARGETS gtest_main gtest)
  endif()

  if(BUILD_TESTING)
    list(APPEND LIBDISTDL_TARGETS distdl_cpp_api_testexe)
    list(APPEND LIBDISTDL_TARGETS distdl_testexe)
  endif(BUILD_TESTING)

  install(
    TARGETS ${LIBDISTDL_TARGETS}
    COMPONENT distdl_cpp_all
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

  add_custom_target(
    install_distdl_cpp
    COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=distdl_cpp_all
            -DCMAKE_INSTALL_PREFIX="${LIBDISTDL_DIR}" -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    DEPENDS distdl_cpp)
  if(BUILD_TESTING)
    add_dependencies(install_distdl_cpp distdl_cpp_api_testexe distdl_testexe)
  endif(BUILD_TESTING)
  add_dependencies(distdl_include_copy install_distdl_cpp)

  string(TOLOWER ${CMAKE_SYSTEM_NAME} CPACK_SYSTEM_NAME)
  set(CPACK_GENERATOR ZIP)
  set(CPACK_PACKAGE_DIRECTORY ${PROJECT_BINARY_DIR}/cpack)
  set(CPACK_PACKAGE_NAME libdistdl)
  # TODO: unify python and c++ version genenerating and getting
  set(CPACK_PACKAGE_VERSION ${DISTDL_CURRENT_VERSION})
  set(CPACK_INSTALL_CMAKE_PROJECTS ${PROJECT_BINARY_DIR};distdl;distdl_cpp_all;/)
  include(CPack)
endif(BUILD_CPP_API)
