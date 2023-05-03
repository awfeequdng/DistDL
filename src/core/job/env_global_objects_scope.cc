
#include "src/core/job/env_global_objects_scope.hh"

#include <thread>
#include <iostream>

#include <glog/logging.h>

namespace distdl {

namespace {

// std::string LogDir(const std::string& log_dir) {
//   static int log_cnt = 0;
//   std::string v = log_dir + std::to_string(log_cnt);
//   log_cnt++;
//   return v;
// }

void InitLogging() {
    google::InitGoogleLogging("distdl");
}

// int32_t GetDefaultCpuDeviceNum() { return std::thread::hardware_concurrency(); }

}  // namespace

EnvGlobalObjectsScope::EnvGlobalObjectsScope(const std::string& env_proto_str) {
    std::cout << "env_proto_str: " << env_proto_str << std::endl;
    Init();
}

EnvGlobalObjectsScope::EnvGlobalObjectsScope() {
    Init();
}

void EnvGlobalObjectsScope::Init() {
    InitLogging();
}

EnvGlobalObjectsScope::~EnvGlobalObjectsScope() {
    VLOG(2) << "Try to close env global objects scope." << std::endl;
    std::cout << "Finish closing env global objects scope." << std::endl;

    google::ShutdownGoogleLogging();
}

EnvGlobalObjectsScope env_global_object_scope;

}  // namespace oneflow
