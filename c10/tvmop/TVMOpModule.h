#ifndef TVM_OP_MODULE_H
#define TVM_OP_MODULE_H
#include <c10/macros/Macros.h>
#include <mutex>

namespace tvm {
namespace runtime {

class Module;
class TVMArgs;
class TVMRetValue;
class C10_API TVMOpModule {
 public:
  // Load TVM operators binary
  void Load(const std::string& filepath);

  void Call(const std::string& fname, const TVMArgs& args, TVMRetValue* rv);

  static TVMOpModule* Get();

  private:
  std::mutex mutex_;
  std::shared_ptr<Module> module_ptr_;
};

}}
#endif // TVM_OP_MODULE_H
