cmake_minimum_required(VERSION 3.20)

project(DistDL LANGUAGES C CXX ASM)

# If turned off: e.g. when ENABLE_FOO is ON, but FOO tool was not found, the CMake will continue.
option(FAIL_ON_UNSUPPORTED_OPTIONS_COMBINATION
   "Stop/Fail CMake configuration if some ENABLE_XXX option is defined (either ON or OFF)
   but is not possible to satisfy" ON)

if(FAIL_ON_UNSUPPORTED_OPTIONS_COMBINATION)
    set(RECONFIGURE_MESSAGE_LEVEL FATAL_ERROR)
else()
    set(RECONFIGURE_MESSAGE_LEVEL WARNING)
endif()

include (cmake/arch.cmake)
include (cmake/target.cmake)
include (cmake/tools.cmake)
include (cmake/ccache.cmake)
include (cmake/clang_tidy.cmake)
include (cmake/git.cmake)

# Ignore export() since we don't use it,
# but it gets broken with a global targets via link_libraries()
macro (export)
endmacro ()

set(CMAKE_EXPORT_COMPILE_COMMANDS 1) # Write compile_commands.json
set(CMAKE_LINK_DEPENDS_NO_SHARED 1) # Do not relink all depended targets on .so
set(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo;Debug;Release;MinSizeRel" CACHE STRING "" FORCE)
# set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Generate debug library name with a postfix.")    # To be consistent with CMakeLists from contrib libs.

# Check that submodules are present
if (NOT EXISTS "${DistDL_SOURCE_DIR}/contrib/sysroot/README.md")
    message (FATAL_ERROR "Submodules are not initialized. Run\n\tgit submodule update --init")
endif ()

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "None")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo")
    message (STATUS "CMAKE_BUILD_TYPE is not set, set to default = ${CMAKE_BUILD_TYPE}")
endif ()
message (STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

string (TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UC)

# list(REVERSE CMAKE_FIND_LIBRARY_SUFFIXES)

# Global libraries
# See:
# - default_libs.cmake
add_library(global-libs INTERFACE)


option(ENABLE_COLORED_BUILD "Enable colors in compiler output" ON)

set (CMAKE_COLOR_MAKEFILE ${ENABLE_COLORED_BUILD}) # works only for the makefile generator

if (ENABLE_COLORED_BUILD AND CMAKE_GENERATOR STREQUAL "Ninja")
    # Turn on colored output. https://github.com/ninja-build/ninja/wiki/FAQ
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")
    # ... such manually setting of flags can be removed once CMake supports a variable to
    # activate colors in *all* build systems: https://gitlab.kitware.com/cmake/cmake/-/issues/15502
    # --> available since CMake 3.24: https://stackoverflow.com/a/73349744
endif ()

include (cmake/check_flags.cmake)
include (cmake/add_warning.cmake)

if (COMPILER_CLANG)
    # generate ranges for fast "addr2line" search
    if (NOT CMAKE_BUILD_TYPE_UC STREQUAL "RELEASE")
        # NOTE: that clang has a bug because of it does not emit .debug_aranges
        # with ThinLTO, so custom ld.lld wrapper is shipped in docker images.
        set(COMPILER_FLAGS "${COMPILER_FLAGS} -gdwarf-aranges")
    endif ()

    # See https://blog.llvm.org/posts/2021-04-05-constructor-homing-for-debug-info/
    if (CMAKE_BUILD_TYPE_UC STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE_UC STREQUAL "RELWITHDEBINFO")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fuse-ctor-homing")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xclang -fuse-ctor-homing")
    endif()

    no_warning(enum-constexpr-conversion) # breaks Protobuf in clang-16
endif ()

option(ENABLE_TESTS "Provide unit_test_DistDL target with Google.Test unit tests" ON)

# if (OS_LINUX AND (ARCH_AMD64 OR ARCH_AARCH64))
#     # Only for Linux, x86_64 or aarch64.
#     option(GLIBC_COMPATIBILITY "Enable compatibility with older glibc libraries." ON)
# elseif(GLIBC_COMPATIBILITY)
#     message (${RECONFIGURE_MESSAGE_LEVEL} "Glibc compatibility cannot be enabled in current configuration")
# endif ()

set(WHOLE_ARCHIVE --whole-archive)
set(NO_WHOLE_ARCHIVE --no-whole-archive)

if (CMAKE_BUILD_TYPE_UC STREQUAL "RELEASE"
    OR CMAKE_BUILD_TYPE_UC STREQUAL "RELWITHDEBINFO"
    OR CMAKE_BUILD_TYPE_UC STREQUAL "MINSIZEREL")
    set (OMIT_HEAVY_DEBUG_SYMBOLS_DEFAULT ON)
else()
    set (OMIT_HEAVY_DEBUG_SYMBOLS_DEFAULT OFF)
endif()
# Provides faster linking and lower binary size.
# Tradeoff is the inability to debug some source files with e.g. gdb
# (empty stack frames and no local variables)."
option(OMIT_HEAVY_DEBUG_SYMBOLS
    "Do not generate debugger info for heavy modules (ClickHouse functions and dictionaries, some contrib)"
    ${OMIT_HEAVY_DEBUG_SYMBOLS_DEFAULT})

if (CMAKE_BUILD_TYPE_UC STREQUAL "DEBUG")
    set(USE_DEBUG_HELPERS ON)
endif()
option(USE_DEBUG_HELPERS "Enable debug helpers" ${USE_DEBUG_HELPERS})

# Optionally split binaries and debug symbols.
option(SPLIT_DEBUG_SYMBOLS "Split binaries and debug symbols" OFF)
if (SPLIT_DEBUG_SYMBOLS)
    message(STATUS "Will split binaries and debug symbols")
    set(SPLITTED_DEBUG_SYMBOLS_DIR "stripped" CACHE STRING "A separate directory for stripped information")
endif()

cmake_host_system_information(RESULT AVAILABLE_PHYSICAL_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY) # Not available under freebsd

if(NOT AVAILABLE_PHYSICAL_MEMORY OR AVAILABLE_PHYSICAL_MEMORY GREATER 8000)
    # Less `/tmp` usage, more RAM usage.
    option(COMPILER_PIPE "-pipe compiler option" ON)
endif()

if(COMPILER_PIPE)
    set(COMPILER_FLAGS "${COMPILER_FLAGS} -pipe")
else()
    message(STATUS "Disabling compiler -pipe option (have only ${AVAILABLE_PHYSICAL_MEMORY} mb of memory)")
endif()

include(cmake/cpu_features.cmake)

# Asynchronous unwind tables
# They are already by default on some platforms but possibly not on all platforms.
# Enable it explicitly.
set (COMPILER_FLAGS "${COMPILER_FLAGS} -fasynchronous-unwind-tables")

# Reproducible builds.
if (CMAKE_BUILD_TYPE_UC STREQUAL "DEBUG")
    set (ENABLE_BUILD_PATH_MAPPING_DEFAULT OFF)
else ()
    set (ENABLE_BUILD_PATH_MAPPING_DEFAULT ON)
endif ()

option (ENABLE_BUILD_PATH_MAPPING "Enable remapping of file source paths in debug info, predefined preprocessor macros, and __builtin_FILE(). It's used to generate reproducible builds. See https://reproducible-builds.org/docs/build-path" ${ENABLE_BUILD_PATH_MAPPING_DEFAULT})

if (ENABLE_BUILD_PATH_MAPPING)
    set (COMPILER_FLAGS "${COMPILER_FLAGS} -ffile-prefix-map=${CMAKE_SOURCE_DIR}=.")
    set (CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -ffile-prefix-map=${CMAKE_SOURCE_DIR}=.")
endif ()


option (ENABLE_BUILD_PROFILING "Enable profiling of build time" OFF)
if (ENABLE_BUILD_PROFILING)
     if (COMPILER_CLANG)
        set (COMPILER_FLAGS "${COMPILER_FLAGS} -ftime-trace")
     else ()
        message (${RECONFIGURE_MESSAGE_LEVEL} "Build profiling is only available with CLang")
     endif ()
endif ()

set (CMAKE_CXX_STANDARD 23)
set (CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_C_STANDARD 11)
set (CMAKE_C_EXTENSIONS ON) # required by most contribs written in C
set (CMAKE_C_STANDARD_REQUIRED ON)

# Compiler-specific coverage flags e.g. -fcoverage-mapping for gcc
option(WITH_COVERAGE "Profile the resulting binary/binaries" OFF)

if (COMPILER_CLANG)
    # Enable C++14 sized global deallocation functions. It should be enabled by setting -std=c++14 but I'm not sure.
    # See https://reviews.llvm.org/D112921
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsized-deallocation")

    # falign-functions=32 prevents from random performance regressions with the code change. Thus, providing more stable
    # benchmarks.
    set(COMPILER_FLAGS "${COMPILER_FLAGS} -falign-functions=32")

    if (ARCH_AMD64)
        # align branches within a 32-Byte boundary to avoid the potential performance loss when code layout change,
        # which makes benchmark results more stable.
        set(BRANCHES_WITHIN_32B_BOUNDARIES "-mbranches-within-32B-boundaries")
        set(COMPILER_FLAGS "${COMPILER_FLAGS} ${BRANCHES_WITHIN_32B_BOUNDARIES}")
    endif()

    if (WITH_COVERAGE)
        set(COMPILER_FLAGS "${COMPILER_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        # If we want to disable coverage for specific translation units
        set(WITHOUT_COVERAGE "-fno-profile-instr-generate -fno-coverage-mapping")
    endif()
endif ()

set (COMPILER_FLAGS "${COMPILER_FLAGS}")

# Our built-in unwinder only supports DWARF version up to 4.
set (DEBUG_INFO_FLAGS "-g -gdwarf-4")

set (CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} ${COMPILER_FLAGS}")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 ${DEBUG_INFO_FLAGS} ${CMAKE_CXX_FLAGS_ADD}")
set (CMAKE_CXX_FLAGS_DEBUG               "${CMAKE_CXX_FLAGS_DEBUG} -O0 ${DEBUG_INFO_FLAGS} ${CMAKE_CXX_FLAGS_ADD}")

set (CMAKE_C_FLAGS                       "${CMAKE_C_FLAGS} ${COMPILER_FLAGS} ${CMAKE_C_FLAGS_ADD}")
set (CMAKE_C_FLAGS_RELWITHDEBINFO        "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 ${DEBUG_INFO_FLAGS} ${CMAKE_C_FLAGS_ADD}")
set (CMAKE_C_FLAGS_DEBUG                 "${CMAKE_C_FLAGS_DEBUG} -O0 ${DEBUG_INFO_FLAGS} ${CMAKE_C_FLAGS_ADD}")

set (CMAKE_ASM_FLAGS                     "${CMAKE_ASM_FLAGS} ${COMPILER_FLAGS} ${CMAKE_ASM_FLAGS_ADD}")
set (CMAKE_ASM_FLAGS_RELWITHDEBINFO      "${CMAKE_ASM_FLAGS_RELWITHDEBINFO} -O3 ${DEBUG_INFO_FLAGS} ${CMAKE_ASM_FLAGS_ADD}")
set (CMAKE_ASM_FLAGS_DEBUG               "${CMAKE_ASM_FLAGS_DEBUG} -O0 ${DEBUG_INFO_FLAGS} ${CMAKE_ASM_FLAGS_ADD}")


if (COMPILER_CLANG)

    # Display absolute paths in error messages. Otherwise KDevelop fails to navigate to correct file and opens a new file instead.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-absolute-paths")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-absolute-paths")

    if (NOT ENABLE_TESTS AND NOT SANITIZE)
        # https://clang.llvm.org/docs/ThinLTO.html
        # Applies to clang only.
        # Disabled when building with tests or sanitizers.
        option(ENABLE_THINLTO "Clang-specific link time optimization" ON)
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-vtable-pointers")

    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
        # Set new experimental pass manager, it's a performance, build time and binary size win.
        # Can be removed after https://reviews.llvm.org/D66490 merged and released to at least two versions of clang.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexperimental-new-pass-manager")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexperimental-new-pass-manager")
    endif ()

    # We cannot afford to use LTO when compiling unit tests, and it's not enough
    # to only supply -fno-lto at the final linking stage. So we disable it
    # completely.
    if (ENABLE_THINLTO AND NOT ENABLE_TESTS AND NOT SANITIZE)
        # Link time optimization
        set (CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -flto=thin -fwhole-program-vtables")
        set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -flto=thin -fwhole-program-vtables")
        set (CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -flto=thin -fwhole-program-vtables")
    elseif (ENABLE_THINLTO)
        message (${RECONFIGURE_MESSAGE_LEVEL} "Cannot enable ThinLTO")
    endif ()

elseif (ENABLE_THINLTO)
    message (${RECONFIGURE_MESSAGE_LEVEL} "ThinLTO is only available with Clang")
endif ()

# Turns on all external libs like s3, kafka, ODBC, ...
option(ENABLE_LIBRARIES "Enable all external libraries by default" ON)

include(cmake/distdl_glob_sources.cmake)


set(BUILD_SHARED_LIBS ON)

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(global-group INTERFACE)
if (OS_LINUX OR OS_ANDROID)
    include(cmake/linux/default_libs.cmake)
else()
    link_libraries(global-group)
endif ()

option(WERROR "Enable -Werror compiler option" ON)

if (WERROR)
    # Don't pollute CMAKE_CXX_FLAGS with -Werror as it will break some CMake checks.
    # Instead, adopt modern cmake usage requirement.
    # TODO: Set CMAKE_COMPILE_WARNING_AS_ERROR (cmake 3.24)
    target_compile_options(global-group INTERFACE "-Werror")
endif ()

# Make this extra-checks for correct library dependencies.
if (OS_LINUX AND NOT SANITIZE)
    target_link_options(global-group INTERFACE "LINKER:--no-undefined")
endif ()

if (OS_LINUX AND NOT (ARCH_AARCH64 OR ARCH_S390X))
    # Slightly more efficient code can be generated
    # It's disabled for ARM because otherwise ClickHouse cannot run on Android.
    set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-pie")
    set (CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fno-pie")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie -Wl,-no-pie")
endif ()

if (ENABLE_TESTS)
    message (STATUS "Unit tests are enabled")
else()
    message(STATUS "Unit tests are disabled")
endif ()

enable_testing() # Enable for tests without binary

option(ENABLE_OPENSSL "This option performs a build with OpenSSL. NOTE! This option is insecure and should never be used. By default, ClickHouse uses and only supports BoringSSL" OFF)

if (ARCH_S390X)
    set(ENABLE_OPENSSL_DYNAMIC_DEFAULT ON)
else ()
    set(ENABLE_OPENSSL_DYNAMIC_DEFAULT OFF)
endif ()
option(ENABLE_OPENSSL_DYNAMIC "This option removes SSL from DistDL and will link to the OpenSSL version supplied by OS." ${ENABLE_OPENSSL_DYNAMIC_DEFAULT})

# when installing to /usr - place configs to /etc but for /usr/local place to /usr/local/etc
if (CMAKE_INSTALL_PREFIX STREQUAL "/usr")
    set (DISTDL_ETC_DIR "/etc")
else ()
    set (DISTDL_ETC_DIR "${CMAKE_INSTALL_PREFIX}/etc")
endif ()

message (STATUS "Building for: ${CMAKE_SYSTEM} ${CMAKE_SYSTEM_PROCESSOR} ${CMAKE_LIBRARY_ARCHITECTURE}")

include (GNUInstallDirs)

# When testing for memory leaks with Valgrind, don't link tcmalloc or jemalloc.

if (TARGET global-group)
    install (EXPORT global DESTINATION cmake)
endif ()

add_subdirectory (contrib EXCLUDE_FROM_ALL)

if (NOT ENABLE_JEMALLOC)
    message (WARNING "Non default allocator is disabled. This is not recommended for production builds.")
endif ()

# Add as many warnings as possible for our own code.
include (cmake/warnings.cmake)
include (cmake/print_flags.cmake)

include_directories(./)

option(BUILD_PYTHON "build python api" ON)
option(DEFAULT_PYTHON_VERSION "default python version" 3)

include (cmake/util.cmake)

message(STATUS "show variables()")

# SHOW_VARIABLES()
set(DISTDL_PYTHON_DIR "${PROJECT_SOURCE_DIR}/python" CACHE PATH "distdl python src dir")

if (BUILD_PYTHON)
    include (cmake/python.cmake)
    include (cmake/pybind11.cmake)

    set(DISTDL_INCLUDE_DIR "${DISTDL_PYTHON_DIR}/distdl/include")
endif ()

include (cmake/distdl.cmake)

