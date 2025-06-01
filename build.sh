#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input # Sửa --no-input thành --noinput nếu Django báo lỗi
python manage.py migrate # BỎ COMMENT DÒNG NÀY
