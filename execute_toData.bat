@echo off
REM ============================================================
REM execute_toData.bat
REM ============================================================

python toData_v7.py ^
  --input_root "D:\DADES_METEO_PC_PREPROCESSADES" ^
  --output_root "D:\DADES_METEO_PC_TO_DATA" ^
  --gpu_devices "cpu" ^
  --max_workers 6 ^
  --group_by_period "none" ^
  --node_coverage_analysis ^
  --use_metric_pos ^
  --log_transform_pluja ^
  --add_wind_components ^
  --include_year_feature ^
  --add_multiscale ^
  --multiscale_radius_quantile 0.55 ^
  --add_edge_weight ^
  --PC_norm_params "PC_norm_params.json"

pause
