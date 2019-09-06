#ifndef TVM_OP_MODULE_H
#define TVM_OP_MODULE_H
#include <mutex>

namespace tvm {
namespace runtime {

class Module;
class TVMArgs;
class TVMRetValue;
class TVMOpModule {
 public:
  // Load TVM operators binary
  void Load(const std::string& filepath);

  void Call(const std::string& fname, const TVMArgs& args, TVMRetValue* rv);

  static TVMOpModule* Get() {
    static TVMOpModule inst;
    return &inst;
  }

  private:
  std::mutex mutex_;
  std::shared_ptr<Module> module_ptr_;
};

}}
#endif // TVM_OP_MODULE_H
