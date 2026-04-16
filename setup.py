# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py
# https://github.com/facebookresearch/detectron2/blob/main/setup.py
# https://github.com/open-mmlab/mmdetection/blob/master/setup.py
# https://github.com/Oneflow-Inc/libai/blob/main/setup.py
# ------------------------------------------------------------------------------------------------
import os
import glob
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension

def get_extensions():
    # Get the absolute path of the current directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "groundingdino", "models", "GroundingDINO", "csrc")

    # Gather the files (these will still be absolute for now)
    raw_sources = [os.path.join(extensions_dir, "vision.cpp")]
    raw_sources += glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    raw_sources += glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)
   
    # CONVERT TO RELATIVE PATHS 
    # This strips "/home/dains/Desktop/.../" and leaves "groundingdino/models/..."
    sources = list(set([os.path.relpath(s, this_dir) for s in raw_sources]))
    print(f"Compiling sources: {sources}")
    
    extra_compile_args = {"cxx": []}
    define_macros = []

    if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
        print(f"Compiling with CUDA (CUDA_HOME: {CUDA_HOME})")
        extension = CUDAExtension
        define_macros += [
            ("WITH_CUDA", None),
            ("__CUDA_NO_HALF_OPERATORS__", None),
            ("__CUDA_NO_HALF_CONVERSIONS__", None),
        ]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "--expt-relaxed-constexpr",
            
        ]
    else:
        print("Compiling without CUDA")
        return [] # Return empty if you only want GPU builds

    include_dirs = [extensions_dir]
    return [
        extension(
            "groundingdino._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)

