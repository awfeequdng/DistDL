cmake_minimum_required(VERSION 3.5)
execute_process(
  COMMAND git describe --tags --always --dirty=-snapshot
  WORKING_DIRECTORY ${DISTDL_GIT_VERSION_ROOT}
  OUTPUT_VARIABLE GIT_REV
  ERROR_QUIET)
if(("${GIT_REV}" STREQUAL "") OR (NOT BUILD_GIT_VERSION))
  set(GIT_REV "N/A")
else()
  string(STRIP "${GIT_REV}" GIT_REV)
endif()

set(VERSION_FILE_CONTENT
    "namespace distdl {\n\
\n\
const char* GetDistDLGitVersion() {\n\
  return \"${GIT_REV}\";\n\
}\n\
\n\
}\n")

if(EXISTS ${DISTDL_GIT_VERSION_FILE})
  file(READ ${DISTDL_GIT_VERSION_FILE} VERSION_FILE_CONTENT_)
else()
  set(VERSION_FILE_CONTENT_ "")
endif()

if(NOT "${VERSION_FILE_CONTENT}" STREQUAL "${VERSION_FILE_CONTENT_}")
  file(WRITE ${DISTDL_GIT_VERSION_FILE} "${VERSION_FILE_CONTENT}")
endif()
