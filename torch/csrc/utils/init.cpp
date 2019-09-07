#ifdef USE_TVM
#include <c10/tvmop/TVMOpModule.h>
#endif
#include <ATen/core/ivalue.h>
#include <torch/csrc/utils/init.h>
#include <torch/csrc/utils/throughput_benchmark.h>

#include <ATen/native/Convolution.h>

#include <pybind11/functional.h>

namespace torch {
namespace throughput_benchmark {

void initThroughputBenchmarkBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  using namespace torch::throughput_benchmark;
  py::class_<BenchmarkConfig>(m, "BenchmarkConfig")
      .def(py::init<>())
      .def_readwrite(
          "num_calling_threads", &BenchmarkConfig::num_calling_threads)
      .def_readwrite("num_worker_threads", &BenchmarkConfig::num_worker_threads)
      .def_readwrite("num_warmup_iters", &BenchmarkConfig::num_warmup_iters)
      .def_readwrite("num_iters", &BenchmarkConfig::num_iters);

  py::class_<BenchmarkExecutionStats>(m, "BenchmarkExecutionStats")
      .def_readonly("latency_avg_ms", &BenchmarkExecutionStats::latency_avg_ms)
      .def_readonly("num_iters", &BenchmarkExecutionStats::num_iters);

  py::class_<ThroughputBenchmark>(m, "ThroughputBenchmark", py::dynamic_attr())
      .def(py::init<jit::script::Module>())
      .def(py::init<py::object>())
      .def(
          "add_input",
          [](ThroughputBenchmark& self, py::args args, py::kwargs kwargs) {
            self.addInput(std::move(args), std::move(kwargs));
          })
      .def(
          "run_once",
          [](ThroughputBenchmark& self, py::args args, py::kwargs kwargs) {
            // Depending on this being ScriptModule of nn.Module we will release
            // the GIL or not further down in the stack
            return self.runOnce(std::move(args), std::move(kwargs));
          })
      .def("benchmark", [](ThroughputBenchmark& self, BenchmarkConfig config) {
        // The benchmark always runs without the GIL. GIL will be used where
        // needed. This will happen only in the nn.Module mode when manipulating
        // inputs and running actual inference
        AutoNoGIL no_gil_guard;
        return self.benchmark(config);
      });


  m.def("_enable_mkldnn_conv", []() {
    at::native::disable_mkldnn_conv.exchange(false);
  });
  m.def("_disable_mkldnn_conv", []() {
    at::native::disable_mkldnn_conv.exchange(true);
  });
  m.def("_load_tvmops", [](const std::string& path){
#ifdef USE_TVM
    tvm::runtime::TVMOpModule::Get()->Load(path); 
#endif
  });
}

} // namespace throughput_benchmark
} // namespace torch
