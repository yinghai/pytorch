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
from .. import defop, AllTypes

def compute_add(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: A[index] + B[index], name='C')
    s = tvm.create_schedule(C.op)
    return s, A, B, C

@defop(name="vadd", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(1, 6)))
def vadd(dtype, ndim):
    s, A, B, C = compute_add(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].parallel(fused)

    return s, [A, B, C]

