@echo off
cd /d "%~dp0"
call Streamlit\venv_fix\Scripts\activate.bat
python -m streamlit run Streamlit/app.py
pause
