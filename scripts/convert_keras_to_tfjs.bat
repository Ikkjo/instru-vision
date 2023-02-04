@echo off

set keraspath=%1
set jspath=%2
set remove=%3

if not defined keraspath (
    @echo No path specified for keras model!
    exit /b 1
)

if not defined jspath (
    @echo No directory specified for output!
    exit /b 1
)

if not exist %keraspath% (
    @echo No such model at path "%keraspath%"!
    @echo Exiting...
    exit /b 1
)

if not exist %jspath% mkdir %jspath%

if not exist "..\temp\venv\" (

    @echo Creating virtual environment...

    python -m venv "../temp/venv"
)

if not errorlevel 1 (
    
    @echo Activating virtual environment...

    call ..\temp\venv\Scripts\activate.bat

    if not errorlevel 1 (
        @echo Installing tensorflowjs...

        pip install tensorflowjs 1>NUL
        
    ) else (
    @echo Error when activating virtual environment! Exiting...
    exit /b 1
    )
) else (
    @echo Error when creating virtual environment! Exiting...
    exit /b 1
)

@echo Converting keras model to tensorflowjs model...

tensorflowjs_converter --input_format=tf_saved_model --output_node_names='MobilenetV3/Predictions/Reshape_1' --saved_model_tags=serve "%keraspath%" "%jspath%" 2>NUL

@echo Converting done!

@echo Deactivating virtual environment...

call ..\temp\venv\Scripts\deactivate.bat

if defined remove (
    if "%remove%"=="r" (
        @echo Removing temporary files...
        rmdir /s /q ..\temp
    ) else (
        @echo INFO: Keeping temporary files.
    )
)

@echo Done!
