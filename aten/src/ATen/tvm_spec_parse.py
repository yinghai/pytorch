from __future__ import print_function
import re
import yaml
import pprint
import sys
import copy

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def parse_tvm_spec_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=Loader)

def run(paths):
    declarations = {} 
    for path in paths:
        for k, v in parse_tvm_spec_yaml(path).items():
            declarations[k] = v
    return declarations
