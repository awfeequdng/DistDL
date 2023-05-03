"""
Copyright 2023 The DistDL Authors. All rights reserved.
"""

import os
# import sys
# import collections
# import warnings

# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-environment-variables
if "CUDA_MODULE_LOADING" not in os.environ:
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import distdl._distdl_internal as ddl_internal

ddl_internal.test.reg_test()

ddl_internal.InitNumpyCAPI()
