import os
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CUDAExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)
        self.sources = sources


class CUDABuild(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CUDAExtension):
            # 获取 Python 头文件路径
            python_include = (
                subprocess.check_output(["python3-config", "--includes"])
                .decode("utf-8")
                .strip()
                .split()
            )

            # 获取 CUDA 库路径
            cuda_lib_dir = (
                "/usr/local/cuda/lib64"  # 这是默认路径，可能需要根据您的系统进行调整
            )

            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.get_ext_fullpath(ext.name)), exist_ok=True)

            # CUDA 编译命令
            nvcc_command = [
                "nvcc",
                "--shared",
                "--compiler-options",
                "'-fPIC'",
                "-o",
                self.get_ext_fullpath(ext.name),
                *ext.sources,
                "-l",
                "cudart",
                "-L",
                cuda_lib_dir,
            ] + python_include

            print("执行的编译命令：", " ".join(nvcc_command))

            try:
                # 执行编译命令
                subprocess.check_call(nvcc_command)
            except subprocess.CalledProcessError as e:
                print(f"编译失败，错误代码：{e.returncode}")
                print(f"命令输出：{e.output}")
                raise
        else:
            super().build_extension(ext)


setup(
    name="cuda_tsp_solver",
    version="0.1",
    ext_modules=[CUDAExtension("cuda_tsp_solver", ["cuda_exhaustion.cu"])],
    cmdclass={
        "build_ext": CUDABuild,
    },
)
