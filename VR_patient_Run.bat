@echo off
cd /d C:\Users\SeungyunLee\Desktop\VR_patient\VR_patient

start "FASTAPI" cmd /k ""C:\Users\SeungyunLee\anaconda3\python.exe" -m uvicorn backend.main:app --reload"
start "STREAMLIT" cmd /k ""C:\Users\SeungyunLee\anaconda3\python.exe" -m streamlit run frontend\frontend.py"

pause