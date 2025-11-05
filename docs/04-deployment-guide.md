# Frame Pipeline - Deployment Guide

> **Diagrammi di Riferimento**: 
> - [System Architecture](img/high_level_architetture.png)
> - [Component Dependencies](img/component_diagram.png)
> - [Data Flow](img/dataflow_architetture.png)
> - [PlantUML Sources](diagrams/)

## Introduzione

Questa guida fornisce istruzioni dettagliate per il deployment del Frame Pipeline.

> **Riferimento Architettura**: Consulta la [panoramica architetturale](01-architecture-overview.md) per comprendere i componenti del sistema prima del deployment.

## Architetture di Deployment

> **Diagramma Sistema**: ![High Level Architecture](img/high_level_architetture.png)

### 1. Deployment Locale (Development)

#### Configurazione Singola Macchina

![System Architecture](img/high_level_architetture.png)

```
┌─────────────────────────────────────────┐
│           Development Machine           │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────┐   │
│  │   Python    │  │     MATLAB      │   │
│  │  Pipeline   │  │   Processing    │   │
│  │   (Linux)   │  │   (Windows)     │   │
│  └─────────────┘  └─────────────────┘   │
│         │                 │             │
│         └─────────┬───────┘             │
│                   │                     │
│          ┌────────▼──────────┐          │
│          │  Shared Storage   │          │
│          │  (Jobs Directory) │          │
│          └───────────────────┘          │
└─────────────────────────────────────────┘
```

**Setup Instructions:**

```bash
# 1. Clone repository
git clone https://github.com/Media4techAI/FrameAIPipeline.git
cd frame_pipeline

# 2. Setup Python environment
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with local paths

# 4. Verify installation
./run_pipeline.py --help
python3.7 test_boost.py all
```

### 2. Deployment Ibrido (Linux + Windows)

#### Configurazione Multi-Macchina

```
┌─────────────────────┐    Network    ┌─────────────────────┐
│   Linux Server      │◄─────────────►│   Windows Server    │
│                     │               │                     │
│  ┌───────────────┐  │               │  ┌───────────────┐  │
│  │   Python      │  │               │  │   MATLAB      │  │
│  │   Pipeline    │  │               │  │   PowerShell  │  │
│  │   (Steps 3-6) │  │               │  │   (Steps 1-2) │  │
│  └───────────────┘  │               │  └───────────────┘  │
│                     │               │                     │
│  ┌───────────────┐  │               │  ┌───────────────┐  │
│  │   CNN Models  │  │               │  │   MATLAB      │  │
│  │   (TensorFlow)│  │               │  │   Scripts     │  │
│  └───────────────┘  │               │  └───────────────┘  │
└─────────────────────┘               └─────────────────────┘
            │                                     │
            └─────────────┐       ┌───────────────┘
                          │       │
                    ┌─────▼───────▼─────┐
                    │   Shared Storage  │
                    │   (NFS/SMB/Cloud) │
                    │                   │
                    │  ┌─────────────┐  │
                    │  │ Job Folders │  │
                    │  │   Logs      │  │
                    │  │   Results   │  │
                    │  └─────────────┘  │
                    └───────────────────┘
```

**Linux Server Setup:**

```bash
# 1. Install dependencies
sudo apt update
sudo apt install python3.7 python3.7-venv python3.7-dev

# 2. Setup project
git clone <repository-url>
cd frame_pipeline
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure shared storage access
sudo mkdir /mnt/shared_jobs
# Mount network storage (NFS example)
sudo mount -t nfs windows-server:/shared_jobs /mnt/shared_jobs

# 4. Configure environment
cat > .env << EOF
PIPELINE_JOBS_DIR=/mnt/shared_jobs
PIPELINE_DEFAULT_CONFIG=/home/pipeline/configs/production.json
TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0
EOF
```

**Windows Server Setup:**

```powershell
# 1. Setup shared directory
New-Item -ItemType Directory -Path "C:\shared_jobs" -Force
New-SmbShare -Name "shared_jobs" -Path "C:\shared_jobs" -FullAccess "Everyone"

# 2. Install MATLAB (manual installation required)

# 3. Setup PowerShell automation
# Copy matlab_pipeline.ps1 to C:\pipeline\
Set-Location "C:\pipeline"

# 4. Configure automation service
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\pipeline\matlab_pipeline.ps1 -WatchPath C:\shared_jobs"
$trigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName "MATLAB_Pipeline_Monitor" -Action $action -Trigger $trigger
```

**Riferimenti:**
- [Architecture Overview](01-architecture-overview.md)
- [Configuration Guide](02-configuration-guide.md)
- [Filtri e Algoritmi](03-filters-algorithms.md)