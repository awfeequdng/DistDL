
# Check if environment is polluted.
if (NOT "$ENV{CFLAGS}" STREQUAL ""
    OR NOT "$ENV{CXXFLAGS}" STREQUAL ""
    OR NOT "$ENV{LDFLAGS}" STREQUAL ""
    OR CMAKE_C_FLAGS OR CMAKE_CXX_FLAGS OR CMAKE_EXE_LINKER_FLAGS OR CMAKE_MODULE_LINKER_FLAGS
    OR CMAKE_C_FLAGS_INIT OR CMAKE_CXX_FLAGS_INIT OR CMAKE_EXE_LINKER_FLAGS_INIT OR CMAKE_MODULE_LINKER_FLAGS_INIT)

    # if $ENV
    message("CFLAGS: $ENV{CFLAGS}")
    message("CXXFLAGS: $ENV{CXXFLAGS}")
    message("LDFLAGS: $ENV{LDFLAGS}")
    # if *_FLAGS
    message("CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
    message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    message("CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
    message("CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
    message("CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")
    # if *_FLAGS_INIT
    message("CMAKE_C_FLAGS_INIT: ${CMAKE_C_FLAGS_INIT}")
    message("CMAKE_CXX_FLAGS_INIT: ${CMAKE_CXX_FLAGS_INIT}")
    message("CMAKE_EXE_LINKER_FLAGS_INIT: ${CMAKE_EXE_LINKER_FLAGS_INIT}")
    message("CMAKE_MODULE_LINKER_FLAGS_INIT: ${CMAKE_MODULE_LINKER_FLAGS_INIT}")

    message(FATAL_ERROR "
        Some of the variables like CFLAGS, CXXFLAGS, LDFLAGS are not empty.
        It is not possible to build DistDL with custom flags.
        These variables can be set up by previous invocation of some other build tools.
        You should cleanup these variables and start over again.

        Run the `env` command to check the details.
        You will also need to remove the contents of the build directory.

        Note: if you don't like this behavior, you can manually edit the cmake files, but please don't complain to developers.")
endif()

# Default toolchain - this is needed to avoid dependency on OS files.
execute_process(COMMAND uname -s OUTPUT_VARIABLE OS)
execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH)

# By default, prefer clang on Linux
# But note, that you still may change the compiler with -DCMAKE_C_COMPILER/-DCMAKE_CXX_COMPILER.
if (OS MATCHES "Linux"
    AND "$ENV{CC}" STREQUAL ""
    AND "$ENV{CXX}" STREQUAL ""
    AND NOT DEFINED CMAKE_C_COMPILER
    AND NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(CLANG_PATH clang)
    if (CLANG_PATH)
        set(CMAKE_C_COMPILER "clang" CACHE INTERNAL "")
    endif()

    find_program(CLANG_CXX_PATH clang++)
    if (CLANG_CXX_PATH)
        set(CMAKE_CXX_COMPILER "clang++" CACHE INTERNAL "")
    endif()
endif()

# if (OS MATCHES "Linux"
#     AND NOT DEFINED CMAKE_TOOLCHAIN_FILE
#     AND NOT DISABLE_HERMETIC_BUILD
#     AND ("$ENV{CC}" MATCHES ".*clang.*" OR CMAKE_C_COMPILER MATCHES ".*clang.*"))

#     if (ARCH MATCHES "amd64|x86_64")
#         set (CMAKE_TOOLCHAIN_FILE "cmake/linux/toolchain-x86_64.cmake" CACHE INTERNAL "")
#     elseif (ARCH MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
#         set (CMAKE_TOOLCHAIN_FILE "cmake/linux/toolchain-aarch64.cmake" CACHE INTERNAL "")
#     elseif (ARCH MATCHES "^(ppc64le.*|PPC64LE.*)")
#         set (CMAKE_TOOLCHAIN_FILE "cmake/linux/toolchain-ppc64le.cmake" CACHE INTERNAL "")
#     elseif (ARCH MATCHES "^(s390x.*|S390X.*)")
#         set (CMAKE_TOOLCHAIN_FILE "cmake/linux/toolchain-s390x.cmake" CACHE INTERNAL "")
#     else ()
#         message (FATAL_ERROR "Unsupported architecture: ${ARCH}")
#     endif ()

# endif()