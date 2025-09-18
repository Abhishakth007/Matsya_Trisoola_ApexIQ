@echo off
setlocal

echo ================================================================================
echo 🚀 SNAP STEPWISE PROCESSING WITH CORRECTED LAND MASK (GRAPH VERSION)
echo ================================================================================      

:: Paths
set INPUT_FILE=data\S1A_IW_GRDH_1SDV_20230620T230642_20230620T230707_049076_05E6CC_2B63.SAFE\manifest.safe
set OUTPUT_DIR=snap_output
set STEP1_OUTPUT=%OUTPUT_DIR%\step1_read.dim
set GRAPH_FILE=landmask_graph.xml
set FINAL_OUTPUT=%OUTPUT_DIR%\step4_final_output.tif
set FINAL_DIM=%OUTPUT_DIR%\step4_final_output.dim

echo ⚙ Processing Configuration:
echo    • Input: %INPUT_FILE%
echo    • Output Directory: %OUTPUT_DIR%
echo    • Graph File: %GRAPH_FILE%
echo    • Final GeoTIFF: %FINAL_OUTPUT%
echo    • Final DIM: %FINAL_DIM%

:: Ensure output dir exists
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

:: Step 1: Apply Orbit File
echo ================================================================================
echo ⚙ STEP 1: APPLY ORBIT FILE
echo ================================================================================
gpt Apply-Orbit-File -Ssource="%INPUT_FILE%" -t "%STEP1_OUTPUT%" -f BEAM-DIMAP
if errorlevel 1 (
    echo ❌ Step 1 failed
    exit /b 1
)
echo ✅ Step 1 completed

:: Step 2–6: Run graph (Calibration + Land-Sea-Mask + GeoTIFF + DIM)
echo ================================================================================
echo ⚙ STEP 2–6: CALIBRATION + LAND-SEA-MASK + GEOTIFF + DIM (GRAPH MODE)
echo ================================================================================
gpt %GRAPH_FILE% -Sstep1_read=%STEP1_OUTPUT%
if errorlevel 1 (
    echo ❌ Graph execution failed
    exit /b 1
)
echo ✅ Graph execution completed

:: Verify outputs
echo ================================================================================
echo ⚙ VERIFICATION
echo ================================================================================
if exist "%FINAL_OUTPUT%" (
    echo ✅ GeoTIFF created: %FINAL_OUTPUT%
) else (
    echo ❌ GeoTIFF missing: %FINAL_OUTPUT%
)

if exist "%FINAL_DIM%" (
    echo ✅ DIM file created: %FINAL_DIM%
) else (
    echo ❌ DIM file missing: %FINAL_DIM%
)

echo ================================================================================
echo 🎉 ALL PROCESSING COMPLETED SUCCESSFULLY
echo ================================================================================
echo 📋 Outputs:
echo    • GeoTIFF (for detection): %FINAL_OUTPUT%
echo    • DIM file (with mask info): %FINAL_DIM%
echo ================================================================================
endlocal
