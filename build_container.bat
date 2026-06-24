@echo off
setlocal enabledelayedexpansion

REM Optional overrides (same folder):
REM   build_env.bat can set IMAGE, TAG, PUSH, DOCKER_USERNAME, DOCKER_PASSWORD
set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%build_env.bat" (
  call "%SCRIPT_DIR%build_env.bat"
)

REM Usage:
REM   build_container.bat [image] [tag] [push]
REM Examples:
REM   build_container.bat
REM   build_container.bat gamemor1/gigaam_node latest push

set "ARG_IMAGE=%~1"
set "ARG_TAG=%~2"
set "ARG_PUSH=%~3"

if not "%ARG_IMAGE%"=="" (
  set "IMAGE=%ARG_IMAGE%"
) else if "%IMAGE%"=="" (
  set "IMAGE=gamemor1/gigaam_node"
)

set "TAG=%ARG_TAG%"
if "%TAG%"=="" set "TAG=latest"

set "DO_PUSH=%ARG_PUSH%"
if "%DO_PUSH%"=="" (
  if "%PUSH%"=="0" (
    set "DO_PUSH="
  ) else (
    set "DO_PUSH=push"
  )
)

echo.
echo Target image: %IMAGE%:%TAG%
if /I "%DO_PUSH%"=="push" (
  echo Push: enabled ^(active docker login session^)
) else (
  echo Push: disabled
)
echo.

if /I "%DO_PUSH%"=="push" (
  if not "%DOCKER_USERNAME%"=="" (
    if not "%DOCKER_PASSWORD%"=="" (
      echo Logging in as %DOCKER_USERNAME% ...
      echo %DOCKER_PASSWORD%| docker login -u "%DOCKER_USERNAME%" --password-stdin
      if errorlevel 1 (
        echo ERROR: docker login failed.
        exit /b 1
      )
    )
  )
)

echo.
echo Building Docker image: %IMAGE%:%TAG%
echo Dockerfile: %CD%\Dockerfile
echo.

docker build -t "%IMAGE%:%TAG%" .
if errorlevel 1 (
  echo ERROR: docker build failed.
  exit /b 1
)

echo.
echo OK: built %IMAGE%:%TAG%

if /I "%DO_PUSH%"=="push" (
  echo.
  echo Pushing %IMAGE%:%TAG% ...
  docker push "%IMAGE%:%TAG%"
  if errorlevel 1 (
    echo ERROR: docker push failed.
    echo Hint: run `docker login` as the owner of %IMAGE% and retry.
    exit /b 1
  )
  echo OK: pushed %IMAGE%:%TAG%
)

echo.
set "RUNPOD_IMAGE=%IMAGE%:%TAG%"
echo %IMAGE% | findstr /I /B "docker.io/" >nul
if errorlevel 1 (
  echo %IMAGE% | findstr /I "^[^/][^/]*/[^/][^/]*" >nul
  if not errorlevel 1 (
    set "RUNPOD_IMAGE=docker.io/%IMAGE%:%TAG%"
  )
)
echo RunPod imageName: !RUNPOD_IMAGE!
echo RunPod (copy/paste): !RUNPOD_IMAGE!
echo Done.
endlocal
