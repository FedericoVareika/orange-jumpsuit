@echo off
setlocal

:: Create the output directory if it doesn't already exist
if not exist "build\lib" (
    mkdir "build\lib"
)

echo Compiling with cl (MSVC)...

:: --- NEW: Setup vendored paths ---
set "OPENBLAS_DIR=vendor\openblas"
set "INCLUDE_FLAGS=/I"%OPENBLAS_DIR%\include""

:: Setup compilation and linker flags
set "COMMON_FLAGS=/O2 /fp:fast /W4"
set "C_STD=/std:c11"
set "DEFINES=/DOJ_EXPORT"

:: --- UPDATED: Use the exact filename and specify the lib path ---
set "LINKER_FLAGS=libopenblas.lib /LIBPATH:"%OPENBLAS_DIR%\lib""

:: Execute the compiler
:: Added %INCLUDE_FLAGS% before the source files
cl.exe /nologo %C_STD% %COMMON_FLAGS% %INCLUDE_FLAGS% %DEFINES% src\jumpsuit.c /LD /Fe"build\lib\libjumpsuit.dll" /Fo"build\lib\\" /link %LINKER_FLAGS%

:: Catch compilation errors
if %ERRORLEVEL% neq 0 (
    echo Compilation failed!
    exit /b %ERRORLEVEL%
)

:: --- NEW: Copy the required DLL to the output folder for runtime ---
echo Copying OpenBLAS DLL to output directory...
copy /Y "%OPENBLAS_DIR%\bin\libopenblas.dll" "build\lib\" > nul

if %ERRORLEVEL% neq 0 (
    echo Failed to copy libopenblas.dll!
    exit /b %ERRORLEVEL%
)

echo Compilation successful!
endlocal
exit /b 0
