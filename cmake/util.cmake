function(SHOW_VARIABLES)
  get_cmake_property(_variableNames VARIABLES)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

macro(write_file_if_different file_path content)
  if(EXISTS ${file_path})
    file(READ ${file_path} current_content)
    # NOTE: it seems a cmake bug that "content" in this macro is not
    # treated as a variable
    if(NOT (current_content STREQUAL ${content}))
      file(WRITE ${file_path} ${content})
    endif()
  else()
    file(WRITE ${file_path} ${content})
  endif()
endmacro()

macro(copy_all_files_in_dir source_dir dest_dir target)
  find_program(rsync rsync)
  if(rsync)
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND
        ${rsync}
        # NOTE: the trailing slash of source_dir is needed.
        # Reference: https://stackoverflow.com/a/56627246
        ARGS -a --omit-dir-times --no-perms --no-owner --no-group --inplace ${source_dir}/
        ${dest_dir})
  else()
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                                                           ${source_dir} ${dest_dir})
  endif()
endmacro()

function(check_cxx11_abi OUTPUT_VAR)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E echo "#include <string>\n void test(std::string){}\n int main(){}"
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/temp.cpp)
  try_compile(
    COMPILE_SUCCESS ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/temp.cpp
    COMPILE_DEFINITIONS -D_GLIBCXX_USE_CXX11_ABI=1
    COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/temp)
  if(NOT COMPILE_SUCCESS)
    message(FATAL_ERROR "Detecting cxx11 availability failed. Please report to DistDL developers.")
  endif()
  execute_process(COMMAND nm ${CMAKE_CURRENT_BINARY_DIR}/temp COMMAND grep -q cxx11
                  RESULT_VARIABLE RET_CODE)
  if(RET_CODE EQUAL 0)
    set(CXX11_ABI_AVAILABLE ON)
  else()
    set(CXX11_ABI_AVAILABLE OFF)
  endif()
  execute_process(COMMAND rm ${CMAKE_CURRENT_BINARY_DIR}/temp ${CMAKE_CURRENT_BINARY_DIR}/temp.cpp)
  set(${OUTPUT_VAR} ${CXX11_ABI_AVAILABLE} PARENT_SCOPE)
endfunction()

include(CheckCXXCompilerFlag)

function(target_try_compile_option target flag)
  # We cannot check for -Wno-foo as this won't throw a warning so we must check for the -Wfoo option directly
  # http://stackoverflow.com/questions/38785168/cc1plus-unrecognized-command-line-option-warning-on-any-other-warning
  string(REGEX REPLACE "^-Wno-" "-W" checkedFlag ${flag})
  string(REGEX REPLACE "[-=]" "_" varName CXX_FLAG${checkedFlag})
  # Avoid double checks. A compiler will not magically support a flag it did not before
  if(NOT DEFINED ${varName}_SUPPORTED)
    check_cxx_compiler_flag(${checkedFlag} ${varName}_SUPPORTED)
  endif()
  if(${varName}_SUPPORTED)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${flag}>)
    if(BUILD_CUDA)
      if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND "${CMAKE_CUDA_COMPILER_ID}" STREQUAL
                                                         "Clang")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${flag}>)
      endif()
    endif()
  endif()
endfunction()

function(target_try_compile_options target)
  foreach(flag ${ARGN})
    target_try_compile_option(${target} ${flag})
  endforeach()
endfunction()

function(target_treat_warnings_as_errors target)
  if(TREAT_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    if(BUILD_CUDA)
      # Only pass flags when cuda compiler is Clang because cmake handles -Xcompiler incorrectly
      if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Werror>)
      endif()
    endif()

    # TODO: remove it while fixing all deprecated call
    target_try_compile_options(${target} -Wno-error=deprecated-declarations)

    # disable unused-* for different compile mode (maybe unused in cpu.cmake, but used in cuda.cmake)
    target_try_compile_options(
      ${target} -Wno-error=unused-const-variable -Wno-error=unused-variable
      -Wno-error=unused-local-typedefs -Wno-error=unused-private-field
      -Wno-error=unused-lambda-capture)

    # there is some strict-overflow warnings in SRC/user/kernels/ctc_loss_kernel_util.cpp for unknown reason, disable them for now
    target_try_compile_options(${target} -Wno-error=strict-overflow)

    target_try_compile_options(${target} -Wno-error=instantiation-after-specialization)

    # disable for pointer operations of intrusive linked lists
    target_try_compile_options(${target} -Wno-error=array-bounds)

    target_try_compile_options(${target} -Wno-error=comment)

    # disable visibility warnings related to https://github.com/Oneflow-Inc/oneflow/pull/3676.
    target_try_compile_options(${target} -Wno-error=attributes)

    # disable error about XXX has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit
    target_try_compile_options(${target} -Wno-error=weak-vtables)

  endif()
endfunction()


function(set_compile_options_to_distdl_target target)
  target_treat_warnings_as_errors(${target})
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>)
  target_compile_definitions(${target} PRIVATE DISTDL_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
  # the mangled name between `struct X` and `class X` is different in MSVC ABI, remove it while windows is supported (in MSVC/cl or clang-cl)
  target_try_compile_options(${target} -Wno-covered-switch-default)

  set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

  if(BUILD_CUDA)
    if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
      target_compile_options(
        ${target}
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                -Xcompiler
                -Werror=return-type;
                -Wno-deprecated-gpu-targets;
                -Werror
                cross-execution-space-call;
                -Xcudafe
                --diag_suppress=declared_but_not_referenced;
                >)
    elseif("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
      target_compile_options(
        ${target}
        PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:
          -Werror=return-type;
          # Suppress warning from cub library -- marking as system header seems not working for .cuh files
          -Wno-pass-failed;
          >)
    else()
      message(FATAL_ERROR "Unknown CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")
    endif()
    # remove THRUST_IGNORE_CUB_VERSION_CHECK if starting using bundled cub
    target_compile_definitions(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                                                 THRUST_IGNORE_CUB_VERSION_CHECK; >)
  endif()
endfunction()

function(check_variable_defined variable)
  if(NOT DEFINED ${variable})
    message(FATAL_ERROR "Variable ${variable} is not defined")
  endif()
endfunction()

# function(checkDirAndAppendSlash)
#   set(singleValues DIR;OUTPUT)
#   set(prefix ARG)
#   cmake_parse_arguments(PARSE_ARGV 0 ${prefix} "${noValues}" "${singleValues}" "${multiValues}")

#   if("${${prefix}_DIR}" STREQUAL "" OR "${${prefix}_DIR}" STREQUAL "/")
#     message(FATAL_ERROR "emtpy path found: ${${prefix}_DIR}")
#   else()
#     set(${${prefix}_OUTPUT} "${${prefix}_DIR}/" PARENT_SCOPE)
#   endif()

# endfunction()

function(mark_targets_as_system)
  # TODO(daquexian): update this function once https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7308
  # and its following PRs are merged in cmake v3.25.
  foreach(target ${ARGV})
    get_target_property(include_dir ${target} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${target} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                               "${include_dir}")
  endforeach()
endfunction()

