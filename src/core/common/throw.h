#pragma once

#include <glog/logging.h>
#include "src/core/common/preprocessor.h"

namespace distdl {

}  // namespace distdl

#define PRINT_BUG_PROMPT_AND_ABORT() LOG(FATAL) << kOfBugIssueUploadPrompt
