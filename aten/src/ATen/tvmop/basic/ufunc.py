# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
import tvm
import topi
from .. import defop, AllTypes

def compute_add(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: A[index] + B[index], name='C')
    s = tvm.create_schedule(C.op)
    return s, A, B, C


def compute_gather(dtype, dim, ndim):
    input = tvm.placeholder([tvm.var() for _ in range(ndim)], name='input', dtype=dtype)
    index = tvm.placeholder([tvm.var() for _ in range(ndim)], name='input', dtype="int32")

    assert len(index.shape) == len(input.shape)
    def c(*indices):
        indices = list(indices)
        gathered = index(*indices)
        indices[dim] = gathered
        return input(*indices)
    out = tvm.compute(index.shape, c, tag=topi.tag.ELEMWISE)
    s = tvm.create_schedule(out.op)
    return s, input, index, out


# non-strided version of gather, specializing dim=0 and rank=[1,6]
@defop(name="tvm_gather", target="cpu", auto_broadcast=False,
       dtype=AllTypes, ndim=list(range(1, 6)))
def tvm_gather(dtype, ndim):
    s, input, index, output = compute_gather(dtype, 0, ndim)
    return s, [input, index, output]


@defop(name="vadd", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(1, 6)))
def vadd(dtype, ndim):
    s, A, B, C = compute_add(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].parallel(fused)

    return s, [A, B, C]

