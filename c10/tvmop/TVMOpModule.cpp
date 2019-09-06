#include "TVMOpModule.h"
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

void TVMOpModule::Load(const std::string& filepath) {
  static const PackedFunc* f_load = Registry::Get("module._LoadFromFile");
  std::lock_guard<std::mutex> lock(mutex_);
  Module module = (*f_load)(filepath, "");
  module_ptr_ = std::make_shared<Module>();
  *module_ptr_ = module;
}

TVMOpModule* TVMOpModule::Get() {
  static TVMOpModule inst;
  return &inst;
}

void TVMOpModule::Call(
    const std::string& fname,
    const TVMArgs& args,
    TVMRetValue* rv) {
  module_ptr_->GetFunction(fname.c_str(), false).CallPacked(args, rv);
}
}
}
