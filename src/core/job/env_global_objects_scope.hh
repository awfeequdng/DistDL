#pragma once
#include <string>

namespace distdl {

class EnvGlobalObjectsScope final {
 public:
  explicit EnvGlobalObjectsScope(const std::string& env_proto_str);
  explicit EnvGlobalObjectsScope();
  ~EnvGlobalObjectsScope();

  // void init_is_normal_exit(bool is_normal_exit) {
  //   is_normal_exit_ = is_normal_exit;
  // }

 private:
  void Init();

 private:
    // bool is_normal_exit_;
};

}  // namespace distdl

