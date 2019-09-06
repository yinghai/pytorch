# Status
It's just a POC to plumb through the flow. Everything is hardcoded for gather.

# Build
Assuming TVM is installed in your Python env.
```
USE_TVM=ON BUILD_CAFFE2_OPS=OFF USE_QNNPACK=OFF USE_FBGEMM=OFF TVM_PATH=/Users/yinghai/tvm python setup.py install
```
# Test
```
import torch
torch._C._load_tvmops('/Users/yinghai/pytorch/build/caffe2/tvmop.so')
input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
index = torch.tensor([[0, 0], [1, 0]])
dim = 0
out = torch.gather(input, dim, index)
```
# TODO
- [ ] Templated dispathing and codgen for TVM path
- [ ] More specialization (stride and etc)
- [ ] Packaging TVM dependency better
- [ ] Support CUDA



