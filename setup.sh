#!/bin/bash

# Tạo môi trường ảo
python3 -m venv venv

# Kiểm tra nếu thư mục venv không tồn tại hoặc không có python3
if [ ! -f "venv/bin/python" ]; then
    echo "Không tìm thấy môi trường ảo! Hãy tạo bằng 'python3 -m venv venv'"
    exit 1
fi

# Cài đặt các thư viện từ requirements.txt
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Kiểm tra thư mục site-packages
SITE_PACKAGES=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

if [ -z "$SITE_PACKAGES" ]; then
    echo "Không tìm thấy thư mục site-packages!"
    exit 1
fi

echo "Đường dẫn site-packages: $SITE_PACKAGES"

# Sao chép sitecustomize.py vào thư mục site-packages
cp sitecustomize.py "$SITE_PACKAGES"

# Kiểm tra lỗi khi sao chép
if [ $? -ne 0 ]; then
    echo "Lỗi khi sao chép sitecustomize.py!"
    exit 1
fi

echo "sitecustomize.py đã được sao chép vào $SITE_PACKAGES thành công!"
