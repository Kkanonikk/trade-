@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Trade Lab

rem --- ищем рабочий Python (лаунчер py, потом python; заглушка Microsoft Store не подходит)
set "PY="
py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY (
  python -c "import sys" >nul 2>nul && set "PY=python"
)
if not defined PY goto :nopython

if exist .venv\Scripts\python.exe goto :venv_ok
if exist .venv rmdir /s /q .venv
echo Создаю виртуальное окружение...
%PY% -m venv .venv
if errorlevel 1 goto :error
:venv_ok
call .venv\Scripts\activate.bat

if exist .venv\.installed goto :installed
echo Ставлю зависимости (нужен интернет, один раз, 2-5 минут)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
python -m pip install -r requirements-optional.txt
if errorlevel 1 echo Необязательный пакет ripser не установился: топология H1 недоступна, остальное работает.
echo ok> .venv\.installed
:installed

rem --- .env необязателен: ключи для открытых источников не нужны
if not exist .env copy .env.example .env >nul

echo.
echo Запускаю Trade Lab. Браузер откроется сам. Если нет, открой http://localhost:8501
echo Чтобы остановить, закрой это окно.
echo.
start "" http://localhost:8501
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
echo.
echo Приложение остановилось. Если это ошибка, скопируй текст выше и пришли мне.
pause
exit /b 0

:nopython
echo Python не найден (или это заглушка Microsoft Store).
echo Установи Python 3.10+ с https://www.python.org/downloads/ и поставь галочку "Add python.exe to PATH".
pause
exit /b 1

:error
echo.
echo Что-то пошло не так, см. сообщения выше. Скопируй их и пришли мне.
pause
exit /b 1
