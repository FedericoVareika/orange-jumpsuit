import os
import sys
import subprocess
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# The directory containing setup.py
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class CustomBuild(build_ext):
    def run(self):
        # 1. Determine platform and script
        if sys.platform == "win32":
            script = "build_lib.bat"
            lib_name = "libjumpsuit.dll"
        else:
            script = "./build_lib.sh"
            lib_name = "libjumpsuit.so"
            
        script_path = os.path.join(ROOT_DIR, script)

        # 2. Run the build script
        print(f"Running C build script: {script}")
        try:
            subprocess.check_call([script_path], cwd=ROOT_DIR)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Error: C library build failed with exit code {e.returncode}")

        # 3. Move the compiled library into the Python package folder
        # Adjust the source path depending on where your scripts output the library
        source_lib = os.path.join(ROOT_DIR, "build", "lib", lib_name) 
        target_lib = os.path.join(ROOT_DIR, "python", "orange_jumpsuit", lib_name)

        if os.path.exists(source_lib):
            shutil.copy(source_lib, target_lib)
            print(f"Copied {lib_name} to {target_lib}")
        else:
            sys.exit(f"Error: Expected output library {source_lib} not found!")

        # 4. Continue with the standard Python build
        super().run()

setup(
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    # Include the .so/.dll files in the final package
    package_data={"orange_jumpsuit": ["*.so", "*.dll"]},
    include_package_data=True,
    # The dummy extension forces a platform-specific wheel
    ext_modules=[Extension("orange_jumpsuit._dummy", sources=[])],
    cmdclass={"build_ext": CustomBuild},
)
