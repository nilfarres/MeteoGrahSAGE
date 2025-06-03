@echo off
REM ============================================================
REM mapa_preds.bat
REM ============================================================

REM EXECUTAR L'SCRIPT AMB ELS PARÀMETRES QUE ES VULGUI
python mapa_preds.py ^
    --ncfile "C:/Users/nfarres/Documents/TFG/models/exec/predictions_meteographpc_test.nc" ^
    --time 1 ^
    --variable Temp ^
    --interp none ^
    --resol 500 ^
    --maxdist 120 ^
    --output "C:/Users/nfarres/Documents/TFG/models/exec/prediccions-temp/mapa_pred" ^
    --all_times

pause
