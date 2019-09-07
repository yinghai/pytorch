from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import os

def load_tvmops():
    p = os.path.join(os.path.dirname(os.path.realpath(torch.__file__)), 'lib')
    print(p)
    torch._C._load_tvmops(os.path.join(p, 'tvmop.so'))

