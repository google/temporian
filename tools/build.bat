:: Build Temporian for Windows
::
:: Usage example:
::   tools\build.bat
::
:: Requirements:
::   - MSYS2
::   - Python versions installed in "C:\Python<version>" e.g. C:\Python310.
::   - Bazel
::   - Visual Studio (tested with VS2019).

cls
setlocal

set TEMPORIAN_VERSION=0.1.6
set BAZEL=bazel.exe
set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC
set BAZEL_SH=C:\msys64\usr\bin\bash.exe
set BAZEL_FLAGS=--config=windows
set PY_DEPS=six python-dateutil absl-py protobuf pandas matplotlib apache-beam tensorflow

CALL :End2End 39 || goto :error
CALL :End2End 310 || goto :error
CALL :End2End 311 || goto :error

:: In case of error
goto :EOF
:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%

:: Runs the full build+test+pip for a specific version of python.
:End2End
set PYTHON_VERSION=%~1
set PYTHON_DIR=C:/Python%PYTHON_VERSION%
set PYTHON=%PYTHON_DIR%/python.exe
set PYTHON3_BIN_PATH=%PYTHON%
set PYTHON3_LIB_PATH=%PYTHON_DIR%/Lib
CALL :Compile %PYTHON% || goto :error
%PYTHON% tools/assemble_pip_files.py || goto :error
CALL :BuildPipPackage %PYTHON% || goto :error
mkdir dist
copy tmp_package\dist\temporian-%TEMPORIAN_VERSION%-cp%PYTHON_VERSION%-cp%PYTHON_VERSION%-win_amd64.whl dist || goto :error
CALL :TestPipPackage dist\temporian-%TEMPORIAN_VERSION%-cp%PYTHON_VERSION%-cp%PYTHON_VERSION%-win_amd64.whl %PYTHON% || goto :error
EXIT /B 0

:: Compiles project with Bazel.
:Compile
set PYTHON=%~1
%PYTHON% -m pip install %PY_DEPS% || goto :error
%BAZEL% clean --expunge
%BAZEL% build %BAZEL_FLAGS% --repo_env PYTHON_BIN_PATH=%PYTHON% -- //...:all || goto :error
EXIT /B 0

:: Builds the pip package
:BuildPipPackage
set PYTHON=%~1
%PYTHON% -m ensurepip -U || goto :error
%PYTHON% -m pip install pip -U || goto :error
%PYTHON% -m pip install setuptools -U || goto :error
%PYTHON% -m pip install build -U || goto :error
%PYTHON% -m pip install virtualenv -U || goto :error
cd tmp_package
%PYTHON% -m build || goto :error
cd ..
EXIT /B 0

:: Tests the pip package.
:TestPipPackage
set PACKAGE=%~1
set PYTHON=%~2
%PYTHON% -m pip uninstall temporian -y || goto :error
%PYTHON% -m pip install %PACKAGE% || goto :error
%PYTHON% tools/check_install.py || goto :error
%PYTHON% -m pip uninstall temporian -y || goto :error
EXIT /B 0
