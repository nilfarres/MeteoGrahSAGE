@echo off
REM ============================================================
REM mapa_preds.bat
REM ============================================================

REM EXECUTA EL SCRIPT AMB ELS PARÀMETRES QUE VULGUIS
python mapa_preds.py ^
    --ncfile "C:/Users/nfarres/Documents/TFG/models/exec_prova4/predictions_meteographpc_test.nc" ^
    --time 1 ^
    --variable Patm ^
    --interp none ^
    --resol 500 ^
    --maxdist 120 ^
    --output "C:/Users/nfarres/Documents/TFG/models/exec_prova4/prediccions-patm/mapa_pred" ^
    --all_times

REM PAUSA PER VEURE EL RESULTAT O MISSATGES D’ERROR
pause
