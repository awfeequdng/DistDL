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