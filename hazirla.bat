@echo off
chcp 65001 >nul
setlocal

REM ===================================================================
REM  Venv python'u burada ayarla. Bos birakirsan PATH'teki python'u
REM  kullanir -- venv AKTIF degilse "No module named tokenizers" hatasi
REM  alirsin. O zaman asagiya venv'in python.exe yolunu yaz, ornegin:
REM  set PY=C:\Users\ahmet\nex1\Scripts\python.exe
REM ===================================================================
set PY=python

echo ============================================
echo   VERI HAZIRLAMA
echo ============================================
%PY% -c "import sys;print('python :',sys.executable)"
if errorlevel 1 goto nopython

%PY% -c "import tokenizers,datasets,numpy;print('paketler: OK')"
if errorlevel 1 goto nopkg

if not exist tokenizer.json goto notok
if not exist data mkdir data

echo.
echo [1/4] Wikipedia...
%PY% prepare_data.py --source wiki --target 1e9 --out data/wiki.bin --tokenizer tokenizer.json --eos_id 0
if errorlevel 1 goto hata

echo.
echo [2/4] Val seti ayriliyor...
%PY% -c "import os,sys;v='data/val.bin';os.path.exists(v) and sys.exit(0);p='data/wiki.bin';n=10000000;sz=os.path.getsize(p);f=open(p,'rb');f.seek(sz-n);t=f.read(n);f.close();open(v,'wb').write(t);f=open(p,'r+b');f.truncate(sz-n);f.close();print('val ayrildi')"
if errorlevel 1 goto hata

echo.
echo [3/4] Kendi verin...
%PY% prepare_data.py --source local --input trcodingdata/dataset --target 1e8 --out data/kendi.bin --tokenizer tokenizer.json --eos_id 0
if errorlevel 1 goto hata

echo.
echo [4/4] FineWeb - UZUN, saatler alabilir...
%PY% prepare_data.py --source fineweb --target 7e9 --out data/fineweb.bin --tokenizer tokenizer.json --eos_id 0
if errorlevel 1 goto hata

echo.
echo ==========  HEPSI BITTI  ==========
dir data
pause
exit /b 0

:nopython
echo.
echo HATA: python bulunamadi. Venv'i aktif et veya yukaridaki
echo       "set PY=" satirina venv python.exe yolunu yaz.
pause & exit /b 1

:nopkg
echo.
echo HATA: tokenizers/datasets/numpy yok -- YANLIS PYTHON calisiyor.
echo       Venv'i aktif ettigin terminalde "where python" calistir,
echo       cikan yolu yukaridaki "set PY=" satirina yaz.
pause & exit /b 1

:notok
echo.
echo HATA: tokenizer.json bu klasorde yok. Once ASAMA 3'u bitir.
pause & exit /b 1

:hata
echo.
echo ######  HATA - yukaridaki mesaji Claude'a at  ######
echo Duzeltip tekrar calistir, kaldigi yerden devam eder.
pause & exit /b 1
