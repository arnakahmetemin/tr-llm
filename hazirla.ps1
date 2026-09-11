# PowerShell surumu (5.1 dahil calisir)
$ErrorActionPreference = "Continue"
if (-not (Test-Path data)) { New-Item -ItemType Directory data | Out-Null }

function Adim($no, $ad, $blok) {
    Write-Host "`n[$no] $ad" -ForegroundColor Cyan
    & $blok
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nHATA: $ad (kod $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "Duzeltip ayni dosyayi tekrar calistir, kaldigi yerden devam eder."
        Read-Host "Enter'a bas"
        exit 1
    }
}

Adim "1/4" "Wikipedia" { python prepare_data.py --source wiki --target 1e9 --out data/wiki.bin --tokenizer tokenizer.json --eos_id 0 }
Adim "2/4" "Val seti ayriliyor" { python -c "import os,sys;v='data/val.bin';os.path.exists(v) and sys.exit(0);p='data/wiki.bin';n=10000000;sz=os.path.getsize(p);f=open(p,'rb');f.seek(sz-n);t=f.read(n);f.close();open(v,'wb').write(t);f=open(p,'r+b');f.truncate(sz-n);f.close();print('val ayrildi')" }
Adim "3/4" "Kendi verin" { python prepare_data.py --source local --input trcodingdata/dataset --target 1e8 --out data/kendi.bin --tokenizer tokenizer.json --eos_id 0 }
Adim "4/4" "FineWeb (UZUN - saatler)" { python prepare_data.py --source fineweb --target 7e9 --out data/fineweb.bin --tokenizer tokenizer.json --eos_id 0 }

Write-Host "`nHEPSI BITTI" -ForegroundColor Green
Get-ChildItem data
Read-Host "Enter'a bas"
