# Frame Pipeline - Guida Completa di Installazione

## Repository GitHub

Il Frame Pipeline System è distribuito attraverso i seguenti repository GitHub:

1. **frame_pipeline**: [https://github.com/Media4techAI/FrameAIPipeline](https://github.com/Media4techAI/FrameAIPipeline)
2. **GlobalReassembly**: [https://github.com/Media4techAI/JigsawNet](https://github.com/Media4techAI/JigsawNet) (branch: `linux-globalreassembly-port`)
3. **Reassembly2d_Sources**: [https://github.com/Media4techAI/Reassembly2d_Sources](https://github.com/Media4techAI/Reassembly2d_Sources)

## Panoramica del Sistema

Il Frame Pipeline è un sistema completo per il riassemblaggio automatico di frammenti di affreschi che combina:

- **frame_pipeline**: Pipeline Python per preprocessing, CNN boost e coordinamento
- **JigsawNet/GlobalReassembly**: Componente C++/CUDA per riassemblaggio globale 
- **Reassembly2d_Sources**: Algoritmi MATLAB per generazione esempi e allineamenti

## Architettura di Deployment Consigliata

```
┌─────────────────────────────────────────────────────────────┐
│                    PC Windows + WSL2                        │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐    ┌─────────────────────────────┐   │
│  │   Windows Host    │    │       WSL2 Debian           │   │
│  │                   │    │                             │   │
│  │ • NVIDIA GPU      │    │ • frame_pipeline (Python)   │   │
│  │ • CUDA Toolkit    │    │ • JigsawNet C++ (CUDA)      │   │
│  │ • MATLAB Runtime  │    │ • Processing & CNN          │   │
│  │ • PowerShell      │    │                             │   │
│  │ • Reassembly2d    │    │                             │   │
│  └───────────────────┘    └─────────────────────────────┘   │
│           │                           │                     │
│           └──────── Shared Storage ───┘                     │
│              /mnt/c/jobs/ ↔ C:\jobs\                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           PowerShell Automation                     │    │
│  │   • FileSystemWatcher monitoring                    │    │
│  │   • MATLAB script execution                         │    │
│  │   • Job status coordination                         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## PARTE 1: Setup Windows Host

### 1.2 Installazione NVIDIA Driver e CUDA

```powershell
# Verifica driver GPU esistente
nvidia-smi

# Se necessario, scarica e installa:
# 1. NVIDIA Game Ready Driver (ultima versione)
#    https://www.nvidia.com/Download/index.aspx

# 2. CUDA Toolkit 11.8 o 12.x
#    https://developer.nvidia.com/cuda-downloads

# Verifica installazione CUDA
nvcc --version
```

### 1.3 Installazione WSL2

```powershell
# Abilita WSL2 (richiede riavvio)
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Riavvia il sistema, poi:
wsl --set-default-version 2

# Installa Debian
wsl --install -d Debian

# Verifica installazione
wsl --list --verbose
```

### 1.4 Installazione MATLAB Runtime

```powershell
# Scarica MATLAB Runtime R2025a
# https://www.mathworks.com/products/compiler/mcr/index.html

# Installa seguendo il wizard
# Percorso predefinito: C:\Program Files\MATLAB\MATLAB Runtime\R2025a\

# Verifica installazione
dir "C:\Program Files\MATLAB\MATLAB Runtime\R2025a\bin\win64"
```

### 1.5 Setup PowerShell Automation

```powershell
# Abilita esecuzione script PowerShell (come Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine

# Verifica policy
Get-ExecutionPolicy

# Crea directory per automation
New-Item -ItemType Directory -Path "C:\pipeline_automation" -Force

# Copia script PowerShell (verrà configurato in seguito)
# Lo script matlab_pipeline.ps1 andrà in questa directory
```

### 1.6 Setup Directory Condivisa e Automazione

```powershell
# Crea directory principale per i job
New-Item -ItemType Directory -Path "C:\pipeline_jobs" -Force
New-Item -ItemType Directory -Path "C:\pipeline_jobs\shared" -Force
New-Item -ItemType Directory -Path "C:\pipeline_jobs\models" -Force
New-Item -ItemType Directory -Path "C:\pipeline_jobs\logs" -Force

# Configura permessi
icacls "C:\pipeline_jobs" /grant Users:F /T

# Copia script PowerShell
# Assicurati che matlab_pipeline.ps1 sia in C:\pipeline_automation\
Copy-Item "path\to\matlab_pipeline.ps1" "C:\pipeline_automation\matlab_pipeline.ps1"

# Copia anche il codice MATLAB Reassembly2d_Sources
Copy-Item "path\to\Reassembly2d_Sources\*" "C:\pipeline_automation\matlab_code\" -Recurse
```

---

## PARTE 2: Setup WSL2 Debian

### 2.1 Accesso e Configurazione Base

```bash
# Entra in WSL2
wsl -d Debian

# Aggiorna sistema
sudo apt update && sudo apt upgrade -y

# Installa dipendenze di base
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    unzip
```

### 2.2 Installazione Python 3.7.9

```bash
# Installa dipendenze Python
sudo apt install -y \
    python3.7 \
    python3.7-dev \
    python3.7-venv \
    python3-pip \
    python3-tk

# Verifica versione
python3.7 --version  # Deve essere 3.7.9

# Crea link simbolico se necessario
sudo ln -sf /usr/bin/python3.7 /usr/bin/python3
```

### 2.3 Installazione CUDA per WSL2

```bash
# Aggiungi repository NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Aggiorna repositories
sudo apt update

# Installa CUDA Toolkit (compatibile con Windows host)
sudo apt install -y cuda-toolkit-11-8

# Configura environment
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verifica installazione
nvcc --version
nvidia-smi  # Deve mostrare GPU Windows
```

### 2.4 Installazione OpenCV e Dipendenze

```bash
# Installa OpenCV
sudo apt install -y \
    libopencv-dev \
    libopencv-contrib-dev \
    python3-opencv

# Installa Eigen3
sudo apt install -y libeigen3-dev

# Installa altre dipendenze C++
sudo apt install -y \
    libomp-dev \
    libboost-all-dev \
    pkg-config
```

---

## PARTE 3: Setup Frame Pipeline

### 3.1 Clone Repository

```bash
# Naviga alla home directory
cd /home/$USER

# Clone repository frame_pipeline
git clone https://github.com/Media4techAI/FrameAIPipeline.git frame_pipeline

# Clone repository JigsawNet (branch specifico)
git clone -b linux-globalreassembly-port https://github.com/Media4techAI/JigsawNet.git JigsawNet

# Clone repository Reassembly2d_Sources
git clone https://github.com/Media4techAI/Reassembly2d_Sources.git Reassembly2d_Sources

# Verifica struttura
ls -la
# Dovresti vedere: frame_pipeline/, JigsawNet/, Reassembly2d_Sources/
```

### 3.2 Setup Frame Pipeline Python

```bash
cd frame_pipeline

# Crea virtual environment
python3.7 -m venv venv
source venv/bin/activate

# Aggiorna pip
pip install --upgrade pip

# Installa dipendenze
pip install -r requirements.txt

# Verifica installazione TensorFlow con GPU
python3.7 -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### 3.3 Configurazione Environment

```bash
# Copia template configurazione
cp .env.example .env

# Edita configurazione
nano .env
```

Contenuto `.env`:
```env
# Directory principale per i job (path Windows accessibile da WSL)
PIPELINE_JOBS_DIR=/mnt/c/pipeline_jobs/shared

# Configurazione predefinita
PIPELINE_DEFAULT_CONFIG=/home/$USER/frame_pipeline/config_template.json

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
TF_CPP_MIN_LOG_LEVEL=1

# Windows PowerShell automation
WINDOWS_JOBS_PATH=C:\pipeline_jobs\shared
POWERSHELL_SCRIPT_PATH=C:\pipeline_automation\matlab_pipeline.ps1

# Logging
LOG_LEVEL=INFO
```

### 3.4 Test Installazione Pipeline

```bash
# Test configurazione
./run_pipeline.sh --help

# Test componenti Python
python3.7 test_boost.py system_info
python3.7 test_boost.py dependencies
python3.7 test_boost.py gpu
```

---

## PARTE 4: Compilazione JigsawNet GlobalReassembly

### 4.1 Preparazione Build

```bash
cd /home/$USER/JigsawNet/GlobalReassembly

# Verifica dipendenze
pkg-config --modversion opencv4
pkg-config --cflags eigen3
nvcc --version
```

### 4.2 Configurazione CMake

```bash
# Crea directory build
mkdir -p build
cd build

# Configura progetto
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86" \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8

# Verifica configurazione
cmake --build . --config Release --dry-run
```

### 4.3 Compilazione

```bash
# Compila progetto
make -j$(nproc)

# Verifica eseguibile
ls -la GlobalReassembly
file GlobalReassembly

# Test base
./GlobalReassembly
```

## PARTE 5: Setup MATLAB Automation (Windows)

### 5.1 Configurazione PowerShell Automation

```powershell
# Su Windows, apri PowerShell come Administrator
cd C:\pipeline_automation

# Verifica script PowerShell
Get-Content matlab_pipeline.ps1 | Select-Object -First 10

# Test syntax
powershell -File matlab_pipeline.ps1 -?
```

### 5.2 Setup MATLAB Runtime Path

```powershell
# Verifica MATLAB Runtime
Test-Path "C:\Program Files\MATLAB\MATLAB Runtime\R2025a\runtime\win64"
Test-Path "C:\Program Files\MATLAB\MATLAB Runtime\R2025a\bin\win64"

# Aggiungi al PATH di sistema se necessario
$oldPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$newPath = $oldPath + ";C:\Program Files\MATLAB\MATLAB Runtime\R2025a\runtime\win64;C:\Program Files\MATLAB\MATLAB Runtime\R2025a\bin\win64"
[Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
```

### 5.3 Test MATLAB Scripts

```powershell
# Test diretto MATLAB
matlab -batch "disp('MATLAB test successful')"

# Test con Reassembly2d code
cd C:\pipeline_automation\matlab_code
matlab -batch "addpath(genpath(pwd)); disp('MATLAB paths configured')"
```

### 5.4 Avvio Automation

```powershell
# Avvia monitoring delle cartelle job
cd C:\pipeline_automation
powershell -File matlab_pipeline.ps1 -WatchPath "C:\pipeline_jobs\shared"

# Per test di un singolo job
powershell -File matlab_pipeline.ps1 -TestJobPath "C:\pipeline_jobs\shared\job_test"
```

---

## PARTE 6: Configurazione Pipeline Completa

### 6.1 Setup Directory Jobs Condivisa

```bash
# Su WSL2 - crea struttura directory (accessibile da Windows)
sudo mkdir -p /mnt/c/pipeline_jobs/
sudo chown -R $USER:$USER /mnt/c/pipeline_jobs/

### 6.2 Configurazione Comunicazione Windows-Linux

```bash
# Su WSL2 - verifica accesso cartella Windows
ls -la /mnt/c/pipeline_jobs/

# Test scrittura
echo "test from WSL2" > /mnt/c/pipeline_jobs/shared/test.txt

# Su Windows PowerShell - verifica lettura
# Get-Content C:\pipeline_jobs\shared\test.txt
```

### 6.3 Test Integrazione Pipeline

**Su Windows (PowerShell):**
```powershell
# Avvia monitoring PowerShell in background
Start-Process -FilePath "powershell" -ArgumentList "-File", "C:\pipeline_automation\matlab_pipeline.ps1", "-WatchPath", "C:\pipeline_jobs\shared" -WindowStyle Minimized
```

**Su WSL2 (Linux):**
```bash
# Attiva ambiente Python
cd /home/$USER/frame_pipeline
source venv/bin/activate

# Test creazione job
./run_pipeline.sh \
    /mnt/c/pipeline_jobs/test_data/masks \
    /mnt/c/pipeline_jobs/test_data/original.jpg \
    "1:9" \
    config_template.json
```

### 6.3 Test Pipeline Completa

```bash
# Test con dati di esempio (se disponibili)
cd /home/$USER/frame_pipeline

# Crea job di test
./run_pipeline.sh \
    /mnt/c/pipeline_jobs/test_data/masks \
    /mnt/c/pipeline_jobs/test_data/original.jpg \
    "1:9" \
    config_template.json

# Monitora esecuzione
tail -f /mnt/c/pipeline_jobs/logs/latest.log
```

---

## PARTE 8: Troubleshooting Comune

### 8.1 Problemi GPU/CUDA

```bash
# Verifica driver NVIDIA
nvidia-smi

# Se non funziona, reinstalla driver WSL2
# Su Windows PowerShell (come Admin):
# wsl --shutdown
# Reinstalla NVIDIA driver che include supporto WSL2

# Verifica CUDA in WSL2
nvcc --version
ls /usr/local/cuda*/bin/
```

### 8.2 Problemi PowerShell/MATLAB

```powershell
# Su Windows - verifica ExecutionPolicy
Get-ExecutionPolicy

# Se necessario, cambia policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Test MATLAB Runtime
matlab -batch "version"

# Verifica percorsi MATLAB
echo $env:PATH | Select-String "MATLAB"

# Test script PowerShell
powershell -File C:\pipeline_automation\matlab_pipeline.ps1 -?
```

**Problemi comuni:**

```powershell
# Se MATLAB non trova runtime:
# Reinstalla MATLAB Runtime R2025a
# Aggiungi manualmente al PATH di sistema

# Se PowerShell blocca script:
Unblock-File C:\pipeline_automation\matlab_pipeline.ps1

# Se monitoring non funziona:
# Verifica permessi cartella
Get-Acl C:\pipeline_jobs\shared
```