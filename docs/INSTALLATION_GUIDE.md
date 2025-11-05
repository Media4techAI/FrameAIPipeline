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

### 1.1 Requisiti Hardware

#### Minimi
- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600 (6+ core)
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA GTX 1660 (6GB VRAM) o superiore
- **Storage**: 50GB SSD liberi
- **OS**: Windows 10 v2004+ / Windows 11

#### Consigliati  
- **CPU**: Intel i7-10700K / AMD Ryzen 7 3700X (8+ core)
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) o superiore 
- **Storage**: 100GB NVMe SSD
- **Network**: Gigabit Ethernet

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

### 4.4 Installazione Globale (Opzionale)

```bash
# Installa eseguibile
sudo cp GlobalReassembly /usr/local/bin/

# Verifica installazione
which GlobalReassembly
GlobalReassembly --help
```

---

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
sudo mkdir -p /mnt/c/pipeline_jobs/shared/{input,output,logs,models,temp}
sudo chown -R $USER:$USER /mnt/c/pipeline_jobs/

# Crea directory modelli
mkdir -p /mnt/c/pipeline_jobs/models/{jigsawnet,boost}
```

### 6.2 Configurazione Comunicazione Windows-Linux

```bash
# Su WSL2 - verifica accesso cartella Windows
ls -la /mnt/c/pipeline_jobs/shared/

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

### 6.2 Configurazione Modelli CNN

```bash
cd /home/$USER/frame_pipeline

# Se hai modelli pre-addestrati, copiali
# cp /path/to/models/* /mnt/c/pipeline_jobs/models/

# Aggiorna percorsi in configurazione
nano frame_pipeline/JigsawCNN/Parameters.py
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

## PARTE 7: Automazione e Monitoring

### 7.1 Script di Health Check

```bash
# Crea script monitoring
cat > /home/$USER/health_check.sh << 'EOF'
#!/bin/bash

echo "=== Frame Pipeline Health Check ==="
echo "Date: $(date)"
echo

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits

# Check Python environment
echo -e "\nPython Environment:"
source /home/$USER/frame_workspace/frame_pipeline/venv/bin/activate
python3.7 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python3.7 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Check CUDA
echo -e "\nCUDA Status:"
nvcc --version | grep "release"

# Check disk space
echo -e "\nDisk Usage:"
df -h /mnt/c/pipeline_jobs

# Check Wine
echo -e "\nWine Status:"
wine --version

echo -e "\n=== Health Check Complete ==="
EOF

chmod +x /home/$USER/health_check.sh
```

### 7.2 Script di Backup

```bash
# Script backup configurazioni
cat > /home/$USER/backup_configs.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/mnt/c/pipeline_jobs/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configurazioni
cp /home/$USER/frame_pipeline/.env "$BACKUP_DIR/"
cp /home/$USER/frame_pipeline/config_template.json "$BACKUP_DIR/"

# Backup modelli (se presenti)
if [ -d "/mnt/c/pipeline_jobs/models" ]; then
    cp -r /mnt/c/pipeline_jobs/models "$BACKUP_DIR/"
fi

echo "Backup completato in: $BACKUP_DIR"
EOF

chmod +x /home/$USER/backup_configs.sh
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

### 8.3 Problemi Python/TensorFlow

```bash
# Ricostruisci virtual environment
cd /home/$USER/frame_pipeline
rm -rf venv
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Test GPU
python3.7 -c "
import tensorflow as tf
print('TF Version:', tf.__version__)
print('GPU Available:', tf.config.list_physical_devices('GPU'))
"
```

### 8.4 Problemi OpenCV/C++

```bash
# Reinstalla OpenCV
sudo apt remove --purge libopencv*
sudo apt autoremove
sudo apt install -y libopencv-dev libopencv-contrib-dev

# Ricompila GlobalReassembly
cd /home/$USER/JigsawNet/GlobalReassembly/build
make clean
cmake ..
make -j$(nproc)
```

---

## PARTE 9: Performance Tuning

### 9.1 Ottimizzazione GPU

```bash
# Configura CUDA per performance
echo 'export CUDA_CACHE_MAXSIZE=2147483648' >> ~/.bashrc  # 2GB cache
echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> ~/.bashrc
source ~/.bashrc
```

### 9.2 Ottimizzazione Memoria

```bash
# Aumenta limite memoria WSL2
# Su Windows, crea %UserProfile%\.wslconfig:
cat > /mnt/c/Users/$USER/.wslconfig << 'EOF'
[wsl2]
memory=24GB
processors=8
swap=8GB
EOF

# Riavvia WSL2
# wsl --shutdown (da Windows PowerShell)
```

### 9.3 Ottimizzazione Storage

```bash
# Mount ottimizzato per performance
sudo nano /etc/fstab
# Aggiungi:
# /mnt/c/pipeline_jobs /pipeline_jobs drvfs defaults,uid=1000,gid=1000,umask=000 0 0
```

---

## PARTE 10: Uso e Test Finale

### 10.1 Verifica Installazione Completa

**Su Windows (PowerShell come Administrator):**
```powershell
# Verifica componenti Windows
Test-Path "C:\Program Files\MATLAB\MATLAB Runtime\R2025a"
Test-Path "C:\pipeline_automation\matlab_pipeline.ps1"
Test-Path "C:\pipeline_jobs\shared"

# Test MATLAB
matlab -batch "disp('MATLAB OK')"
```

**Su WSL2 (Linux):**
```bash
# Health check completo
/home/$USER/health_check.sh

# Test pipeline components
cd /home/$USER/frame_pipeline
source venv/bin/activate
python3.7 test_boost.py system_info
```

### 10.2 Script di Avvio Completo

**Su Windows (PowerShell):**
```powershell
# Crea script startup Windows
@'
# Frame Pipeline Windows Startup
Write-Host "Starting Frame Pipeline Windows Components..." -ForegroundColor Cyan

# Verifica MATLAB Runtime
if (Test-Path "C:\Program Files\MATLAB\MATLAB Runtime\R2025a") {
    Write-Host "✓ MATLAB Runtime found" -ForegroundColor Green
} else {
    Write-Host "✗ MATLAB Runtime not found" -ForegroundColor Red
    exit 1
}

# Avvia PowerShell automation
Write-Host "Starting PowerShell automation..." -ForegroundColor Yellow
Start-Process -FilePath "powershell" -ArgumentList "-File", "C:\pipeline_automation\matlab_pipeline.ps1", "-WatchPath", "C:\pipeline_jobs\shared" -WindowStyle Minimized

Write-Host "Windows components ready!" -ForegroundColor Green
Write-Host "MATLAB automation monitoring: C:\pipeline_jobs\shared" -ForegroundColor Gray
'@ | Out-File -FilePath "C:\pipeline_automation\start_windows.ps1" -Encoding UTF8
```

**Su WSL2 (Linux):**
```bash
# Crea script startup Linux (aggiornato)
cat > /home/$USER/start_pipeline.sh << 'EOF'
#!/bin/bash

echo "Starting Frame Pipeline Linux Components..." -e "\033[36m"

# Verifica comunicazione con Windows
if [ ! -d "/mnt/c/pipeline_jobs/shared" ]; then
    echo "✗ Windows shared directory not accessible" -e "\033[31m"
    exit 1
fi
echo "✓ Windows shared directory accessible" -e "\033[32m"

# Attiva Python environment
cd /home/$USER/frame_pipeline
source venv/bin/activate

# Verifica componenti critici
echo "Checking GPU..."
nvidia-smi --query-gpu=name --format=csv,noheader

echo "Checking Python packages..."
python3.7 -c "import tensorflow as tf; import cv2; print('Ready!')"

echo "Environment ready!" -e "\033[32m"
echo "Usage: ./run_pipeline.sh <masks_dir> <original_image> <parts> [config_file]"
echo "Note: Make sure Windows PowerShell automation is running first"
EOF

chmod +x /home/$USER/start_pipeline.sh
```

### 10.3 Documentazione Custom

```bash
# Crea documentazione specifica installazione
cat > /home/$USER/INSTALLATION_NOTES.md << 'EOF'
# My Frame Pipeline Installation

## System Info
- OS: $(lsb_release -d | cut -f2)
- CUDA: $(nvcc --version | grep release | cut -d' ' -f6)
- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
- Python: $(python3.7 --version)

## Key Paths
- Pipeline: /home/$USER/frame_pipeline
- GlobalReassembly: /home/$USER/JigsawNet/GlobalReassembly/build/GlobalReassembly
- Jobs Directory: /mnt/c/pipeline_jobs/shared
- Models: /mnt/c/pipeline_jobs/models

## Quick Commands
- Health Check: ~/health_check.sh
- Start Environment: ~/start_pipeline.sh
- Backup Configs: ~/backup_configs.sh

## Installation Date
$(date)
EOF
```

---

## Conclusione

Ora hai una installazione completa del Frame Pipeline con:

✅ **Windows Host**: GPU, CUDA, MATLAB Runtime, PowerShell Automation  
✅ **WSL2 Debian**: Python 3.7.9, TensorFlow-GPU, OpenCV  
✅ **Frame Pipeline**: Pipeline Python configurata  
✅ **GlobalReassembly**: Componente C++/CUDA compilato  
✅ **MATLAB Automation**: PowerShell monitoring con comunicazione Windows-Linux  
✅ **Monitoring**: Script di health check e backup  

### Architettura Finale

```
Windows Host (MATLAB + PowerShell)
         ↕ (Shared Directory)
    WSL2 Linux (Python Pipeline)
```

**Flusso di lavoro:**
1. **WSL2**: Crea job in `/mnt/c/pipeline_jobs/shared/`
2. **Windows**: PowerShell monitora `C:\pipeline_jobs\shared\` 
3. **Windows**: Esegue MATLAB scripts (generate_new_example, export_alignments)
4. **WSL2**: Continua con CNN boost e ricostruzione finale

### Prossimi Passi

1. **Testa con dati reali**: Usa i tuoi dataset di frammenti
2. **Ottimizza performance**: Tuning basato sui tuoi workload
3. **Automazione**: Setup di pipeline batch per elaborazioni multiple
4. **Monitoring**: Implementa logging avanzato per produzione

### Supporto

- **Logs Windows**: `C:\pipeline_jobs\logs\`
- **Logs Linux**: `/mnt/c/pipeline_jobs/logs/`
- **Health Check**: Esegui `~/health_check.sh` per diagnostica
- **Backup**: Usa `~/backup_configs.sh` prima di modifiche importanti
- **PowerShell Automation**: Monitora via `C:\pipeline_automation\matlab_pipeline.ps1`
- **Repository GitHub**: Controlla gli aggiornamenti sui repository ufficiali Media4techAI

**Installazione completata con successo!** 🎉