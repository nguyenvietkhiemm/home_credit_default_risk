@echo off
setlocal enabledelayedexpansion



if not exist venv\Scripts\python.exe (
    echo Không tìm thấy môi trường ảo! Hãy tạo bằng 'python -m venv venv'
    exit /b
)

for /d %%i in (venv\Lib\site-packages*) do set SITE_PACKAGES=%%i

if not defined SITE_PACKAGES (
    echo Không tìm thấy thư mục site-packages!
    exit /b
)

echo Đường dẫn site-packages: %SITE_PACKAGES%

copy /Y sitecustomize.py "%SITE_PACKAGES%"

if %errorlevel% neq 0 (
    echo Lỗi khi sao chép sitecustomize.py!
    exit /b
)

echo sitecustomize.py đã được sao chép vào %SITE_PACKAGES% thành công!
pause
