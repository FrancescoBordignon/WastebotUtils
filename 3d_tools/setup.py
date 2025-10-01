from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess

def get_opencv_flags():
    cflags = subprocess.check_output(["pkg-config", "--cflags", "opencv4"]).decode().strip().split()
    libs = subprocess.check_output(["pkg-config", "--libs", "opencv4"]).decode().strip().split()
    return cflags, libs

opencv_cflags, opencv_libs = get_opencv_flags()

# Add optimization flags
optimization_flags = ['-O3', '-flto']

ext_modules = [
    Pybind11Extension(
        "depth_reprojector",
        ["src/bindings.cpp", "src/depth_reprojector_eigen_op1_class.cpp"],
        extra_compile_args=opencv_cflags + optimization_flags,
        extra_link_args=opencv_libs + ['-flto'],  # link time optimization
        include_dirs=[
            "headers",              
            "/usr/include/eigen3",  # Eigen headers
        ],
        language="c++"
    ),
]

setup(
    name="depth_reprojector",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
