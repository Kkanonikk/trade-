@echo off
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python не найден. Установи Python 3.10+ с python.org (галочка "Add to PATH"^) и запусти снова.
  pause
  exit /b 1
)

if not exist .venv (
  echo Создаю виртуальное окружение...
  python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat

if not exist .venv\.installed (
  echo Ставлю зависимости (нужен интернет, один раз^)...
  python -m pip install --upgrade pip >nul
  python -m pip install -r requirements.txt || goto :error
  python -m pip install -r requirements-optional.txt || echo Необязательный пакет ripser не установился: топология H1 будет недоступна, остальное работает.
  echo ok> .venv\.installed
)

if not exist .env (
  copy .env.example .env >nul
  echo Создан файл .env. Открой его, вставь BCS_REFRESH_TOKEN и запусти start.bat снова.
  notepad .env
  pause
  exit /b 0
)

echo Запускаю на http://localhost:8501
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
exit /b 0

:error
echo Что-то пошло не так, см. сообщения выше.
pause
exit /b 1
