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
"""TVM Operator compile entry point"""
import tvm

from collections import OrderedDict
import yaml
import os
import argparse
from tvmop.opdef import __OP_DEF__

def get_target(device):
    if device == "cpu":
        return "llvm"
    elif device == "cuda" or device == "gpu":
        return "cuda"
    assert False, "Unknown device " + device


def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def format_yaml(data):
    noalias_dumper = yaml.dumper.SafeDumper
    noalias_dumper.ignore_aliases = lambda self, data: True
    # Support serializing OrderedDict
    noalias_dumper.add_representer(OrderedDict, dict_representer)
    # Some yaml parsers (e.g. Haskell's) don't understand line breaks.
    # width=float('Inf') turns off optional line breaks and improves
    # the portability of the outputted yaml.
    return yaml.dump(data, default_flow_style=False, Dumper=noalias_dumper, width=float('Inf'))


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))
    parser = argparse.ArgumentParser(description="Generate tvm operators")
    parser.add_argument("-o", action="store", required=True, dest="target_path",
                        help="Target path which stores compiled library")
    parser.add_argument("-t", action="store", required=True, dest="spec_path",
                        help="Spec path which stores the specialization specs")
    arguments = parser.parse_args()

    func_list_llvm = []
    func_list_cuda = []

    # TODO: attach instruction features to the library, e.g., avx-512, etc.
    # Dispatching order:
    # - dytpe
    #  - rank
    #   - integer attrs
    specs = {}
    for operator_def in __OP_DEF__:
        specialization_list = []
        for sch, args, name, kwargs in operator_def.invoke_all():
            if tvm.module.enabled(get_target(operator_def.target)):
                func_list = func_list_llvm if operator_def.target == "cpu" else func_list_cuda
                kwargs['func_name'] = name
                specialization_list.append(kwargs)
                func_lower = tvm.lower(sch, args,
                                       name=name,
                                       binds=operator_def.get_binds(args))
                func_list.append(func_lower)
        properties = {'kwargs' : operator_def.get_kwargs(), 'specialization' : specialization_list}
        specs[operator_def.name] = properties

    lowered_funcs = {get_target("cpu") : func_list_llvm}
    if len(func_list_cuda) > 0:
        lowered_funcs[get_target("cuda")] = func_list_cuda
    func_binary = tvm.build(lowered_funcs, name="tvmop")
    func_binary.export_library(arguments.target_path)
    print("TVM ops written to: {}".format(arguments.target_path))
    with open(arguments.spec_path, 'w') as f:
        f.write(format_yaml(specs))
    print("TVM specs written to: {}".format(arguments.spec_path))
