import os
import sys
import subprocess
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

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

        # 3. Move the compiled library
        source_lib = os.path.join(ROOT_DIR, "build", "lib", lib_name) 

        # --- THE FIX IS HERE ---
        # A. Target for local development (Source Tree)
        source_target_dir = os.path.join(ROOT_DIR, "python", "orange_jumpsuit")
        os.makedirs(source_target_dir, exist_ok=True)
        
        # B. Target for the Wheel package (Build Tree)
        # self.build_lib points to the temporary folder where the wheel is assembled
        wheel_target_dir = os.path.join(self.build_lib, "orange_jumpsuit")
        os.makedirs(wheel_target_dir, exist_ok=True)

        if os.path.exists(source_lib):
            # Copy to both locations
            shutil.copy(source_lib, os.path.join(source_target_dir, lib_name))
            shutil.copy(source_lib, os.path.join(wheel_target_dir, lib_name))
            print(f"Copied {lib_name} to wheel build directory.")
        else:
            sys.exit(f"Error: Expected output library {source_lib} not found!")

        # Do the exact same thing for the OpenBLAS DLL on Windows
        if sys.platform == "win32":
            openblas_dll_name = "libopenblas.dll" 
            source_openblas = os.path.join(ROOT_DIR, "vendor", "openblas", "bin", openblas_dll_name)
            
            if os.path.exists(source_openblas):
                shutil.copy(source_openblas, os.path.join(source_target_dir, openblas_dll_name))
                shutil.copy(source_openblas, os.path.join(wheel_target_dir, openblas_dll_name))
                print(f"Copied {openblas_dll_name} to wheel build directory.")
            else:
                sys.exit(f"Error: Expected OpenBLAS DLL at {source_openblas} not found!")

        # 4. Continue with the standard Python build
        super().run()

setup(
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={"orange_jumpsuit": ["*.so", "*.dll"]},
    include_package_data=True,
    ext_modules=[Extension("orange_jumpsuit._dummy", sources=["python/dummy.c"])],
    cmdclass={"build_ext": CustomBuild},
)
