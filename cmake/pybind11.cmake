include (cmake/python.cmake)

function(pybind11_extension name)
    set_target_properties(${name} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                             SUFFIX "${PYTHON_MODULE_EXTENSION}")
endfunction()

# Build a Python extension module:
# pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
#                     [NO_EXTRAS] [THIN_LTO] [OPT_SIZE] source1 [source2 ...])
#
function(pybind11_add_module target_name)
    set(options "MODULE;SHARED;EXCLUDE_FROM_ALL;NO_EXTRAS;SYSTEM;THIN_LTO;OPT_SIZE")
    cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})

    if(ARG_MODULE AND ARG_SHARED)
        message(FATAL_ERROR "Can't be both MODULE and SHARED")
    elseif(ARG_SHARED)
        set(lib_type SHARED)
    else()
        set(lib_type MODULE)
    endif()

    if(ARG_EXCLUDE_FROM_ALL)
        set(exclude_from_all EXCLUDE_FROM_ALL)
    else()
        set(exclude_from_all "")
    endif()

    add_library(${target_name} ${lib_type} ${exclude_from_all} ${ARG_UNPARSED_ARGUMENTS})

    if(ARG_SYSTEM)
        message(
            STATUS
            "Warning: this does not have an effect - use NO_SYSTEM_FROM_IMPORTED if using imported targets"
        )
    endif()

    pybind11_extension(${target_name})

    # -fvisibility=hidden is required to allow multiple modules compiled against
    # different pybind versions to work properly, and for some features (e.g.
    # py::module_local).  We force it on everything inside the `pybind11`
    # namespace; also turning it on for a pybind module compilation here avoids
    # potential warnings or issues from having mixed hidden/non-hidden types.
    if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
        set_target_properties(${target_name} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
    endif()

    if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
        set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
    endif()

    if(ARG_NO_EXTRAS)
        return()
    endif()

    if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        message(STATUS "not implement lto")
        # if(ARG_THIN_LTO)
        #     target_link_libraries(${target_name} PRIVATE pybind11::thin_lto)
        # else()
        #     target_link_libraries(${target_name} PRIVATE pybind11::lto)
        # endif()
    endif()

    # Use case-insensitive comparison to match the result of $<CONFIG:cfgs>
    string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
        if(NOT MSVC AND NOT "${uppercase_CMAKE_BUILD_TYPE}" MATCHES DEBUG|RELWITHDEBINFO)
        # pybind11_strip(${target_name})
        string(STRIP ${target_name} target_name)
    endif()

    if(ARG_OPT_SIZE)
        message(STATUS "not implement OPT_SIZE")
        # target_link_libraries(${target_name} PRIVATE pybind11::opt_size)
    endif()

    target_link_libraries(${target_name}
        PRIVATE
        ch_contrib::pybind11
        ${Python3_LIBRARIES}
    )

    target_include_directories(${target_name} PRIVATE ${Python_INCLUDE_DIRS})

endfunction()
