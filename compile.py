from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='minference',
    ext_modules=[
        CUDAExtension(
            name='minference',
            sources=['./sparse_frontier/modelling/attention/minference/csrc/kernels.cpp', './sparse_frontier/modelling/attention/minference/csrc/vertical_slash_index.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    package_dir={'': 'sparse_frontier/modelling/attention/minference'}
)
