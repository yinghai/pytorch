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

def compute_gather(dtype, dim, ndim):
    input = tvm.placeholder([tvm.var() for _ in range(ndim)], name='input', dtype=dtype)
    index = tvm.placeholder([tvm.var() for _ in range(ndim)], name='input', dtype="int64")

    assert len(index.shape) == len(input.shape)
    def c(*indices):
        indices = list(indices)
        gathered = index(*indices)
        indices[dim] = gathered
        return input(*indices)
    out = tvm.compute(index.shape, c, tag=topi.tag.ELEMWISE)
    s = tvm.create_schedule(out.op)
    return s, input, index, out


def check_gather_input(dtype, rank, dim):
    return dim < rank

# non-strided version of gather, specializing dim=0 and rank=[1,6]
@defop(name="tvm_gather", target="cpu", auto_broadcast=False,
       dtype=AllTypes, rank=list(range(1, 6)), dim=list(range(0, 6)),
       attrs_valid=check_gather_input, attrs=['dim'])
def tvm_gather(dtype, dim, rank):
    s, input, index, output = compute_gather(dtype, dim, rank)
    return s, [input, index, output]


