from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import os, os.path, subprocess

class BuildRvo2Ext(_build_ext):
    def run(self):
        build_dir = os.path.abspath("build/RVO2")
        lib = os.path.join(build_dir, "src", "RVO.lib")
        if not os.path.exists(lib):
            if not os.path.exists(build_dir):
                os.makedirs(build_dir)
            subprocess.check_call(["cmake", "../..", "-G", "Visual Studio 17 2022", "-A", "x64"], cwd=build_dir)
            subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir)
            release_lib = os.path.join(build_dir, "src", "Release", "RVO.lib")
            if os.path.exists(release_lib):
                import shutil
                os.makedirs(os.path.join(build_dir, "src"), exist_ok=True)
                shutil.copy2(release_lib, lib)
        _build_ext.run(self)

extensions = [
    Extension(
        "rvo2",
        ["src/rvo2.pyx"],
        include_dirs=["src"],
        libraries=["RVO"],
        library_dirs=["build/RVO2/src", "build/RVO2/src/Release", "."],
    ),
]

setup(
    name="pyrvo2",
    ext_modules=cythonize(extensions, language_level=3),
    cmdclass={"build_ext": BuildRvo2Ext},
)
