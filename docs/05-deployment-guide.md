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
git clone <repository-url>
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

## Monitoring e Observability

### 1. Health Checks

```python
# health_check.py
import os
import json
import psutil
from datetime import datetime
from lib.parameters import Parameters

class HealthChecker:
    def __init__(self):
        self.checks = []
        
    def check_system_resources(self):
        """Verifica risorse di sistema"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'disk_usage': disk.percent,
            'available_memory_gb': memory.available / (1024**3)
        }
        
    def check_model_availability(self):
        """Verifica disponibilità modelli CNN"""
        model_paths = [f"model/g{i}/" for i in range(5)]
        models_status = {}
        
        for path in model_paths:
            checkpoint_file = os.path.join(path, "checkpoint")
            models_status[path] = os.path.exists(checkpoint_file)
            
        return models_status
        
    def check_dependencies(self):
        """Verifica dipendenze Python"""
        try:
            import tensorflow as tf
            import cv2
            import numpy as np
            
            return {
                'tensorflow_version': tf.__version__,
                'opencv_available': True,
                'numpy_available': True
            }
        except ImportError as e:
            return {
                'error': str(e),
                'dependencies_ok': False
            }
            
    def run_full_check(self):
        """Esegue check completo"""
        timestamp = datetime.now().isoformat()
        
        health_report = {
            'timestamp': timestamp,
            'system_resources': self.check_system_resources(),
            'model_availability': self.check_model_availability(),
            'dependencies': self.check_dependencies()
        }
        
        # Determina stato generale
        overall_status = self.determine_overall_status(health_report)
        health_report['status'] = overall_status
        
        return health_report
        
if __name__ == "__main__":
    checker = HealthChecker()
    report = checker.run_full_check()
    print(json.dumps(report, indent=2))
```

### 2. Metrics Collection

```python
# metrics_collector.py
import time
import json
import threading
from datetime import datetime, timedelta

class MetricsCollector:
    def __init__(self, output_file="metrics.json"):
        self.output_file = output_file
        self.metrics = []
        self.collection_interval = 60  # seconds
        self.running = False
        
    def collect_pipeline_metrics(self):
        """Raccoglie metriche specifiche della pipeline"""
        
        # Job metrics
        job_metrics = self.collect_job_metrics()
        
        # Performance metrics
        perf_metrics = self.collect_performance_metrics()
        
        # Error metrics
        error_metrics = self.collect_error_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'job_metrics': job_metrics,
            'performance_metrics': perf_metrics,
            'error_metrics': error_metrics
        }
        
    def collect_job_metrics(self):
        """Metriche relative ai job"""
        jobs_dir = os.environ.get('PIPELINE_JOBS_DIR', './jobs')
        
        if not os.path.exists(jobs_dir):
            return {'error': 'Jobs directory not found'}
            
        job_folders = [d for d in os.listdir(jobs_dir) 
                      if os.path.isdir(os.path.join(jobs_dir, d))]
        
        completed_jobs = 0
        failed_jobs = 0
        processing_jobs = 0
        
        for job_folder in job_folders:
            status = self.get_job_status(os.path.join(jobs_dir, job_folder))
            if status == 'completed':
                completed_jobs += 1
            elif status == 'failed':
                failed_jobs += 1
            elif status == 'processing':
                processing_jobs += 1
                
        return {
            'total_jobs': len(job_folders),
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'processing_jobs': processing_jobs
        }
        
    def start_collection(self):
        """Avvia raccolta automatica metriche"""
        self.running = True
        
        def collection_loop():
            while self.running:
                try:
                    metrics = self.collect_pipeline_metrics()
                    self.metrics.append(metrics)
                    self.save_metrics()
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
                    
                time.sleep(self.collection_interval)
                
        self.collection_thread = threading.Thread(target=collection_loop)
        self.collection_thread.start()
```

### 3. Alerting System

```python
# alerting.py
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertManager:
    def __init__(self, config_file="alerting_config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
    def check_alert_conditions(self, metrics):
        """Verifica condizioni di alert"""
        alerts = []
        
        # Check CPU usage
        if metrics['system_resources']['cpu_usage'] > 90:
            alerts.append({
                'severity': 'warning',
                'message': f"High CPU usage: {metrics['system_resources']['cpu_usage']}%"
            })
            
        # Check memory usage
        if metrics['system_resources']['memory_usage'] > 85:
            alerts.append({
                'severity': 'warning', 
                'message': f"High memory usage: {metrics['system_resources']['memory_usage']}%"
            })
            
        # Check failed jobs
        if 'job_metrics' in metrics:
            failed_rate = (metrics['job_metrics']['failed_jobs'] / 
                          max(metrics['job_metrics']['total_jobs'], 1))
            if failed_rate > 0.1:  # 10% failure rate
                alerts.append({
                    'severity': 'critical',
                    'message': f"High job failure rate: {failed_rate:.2%}"
                })
                
        return alerts
        
    def send_alert(self, alert):
        """Invia notifica di alert"""
        if self.config.get('email_enabled', False):
            self.send_email_alert(alert)
            
        if self.config.get('slack_enabled', False):
            self.send_slack_alert(alert)
            
    def send_email_alert(self, alert):
        """Invia alert via email"""
        smtp_config = self.config['email']
        
        msg = MIMEMultipart()
        msg['From'] = smtp_config['from']
        msg['To'] = ', '.join(smtp_config['to'])
        msg['Subject'] = f"Frame Pipeline Alert - {alert['severity'].upper()}"
        
        body = f"""
        Alert Details:
        Severity: {alert['severity']}
        Message: {alert['message']}
        Timestamp: {datetime.now().isoformat()}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
            if smtp_config.get('use_tls', False):
                server.starttls()
            if smtp_config.get('username'):
                server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"Failed to send email alert: {e}")
```

## Security e Best Practices

### 1. Security Configuration

```yaml
# security-config.yml
security:
  authentication:
    enabled: true
    method: "jwt"
    secret_key: "${JWT_SECRET_KEY}"
    
  authorization:
    roles:
      - name: "pipeline_admin"
        permissions: ["create_job", "delete_job", "view_logs", "system_config"]
      - name: "pipeline_user" 
        permissions: ["create_job", "view_own_jobs"]
        
  data_protection:
    encryption_at_rest: true
    encryption_in_transit: true
    key_rotation_days: 90
    
  network:
    allowed_ips:
      - "10.0.0.0/8"
      - "192.168.0.0/16"
    firewall_rules:
      - port: 8080
        protocol: "tcp"
        source: "internal"
        
  audit:
    enabled: true
    log_level: "info"
    retention_days: 365
```

### 2. Backup Strategy

```bash
#!/bin/bash
# backup_pipeline.sh

BACKUP_DIR="/backup/frame_pipeline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="${BACKUP_DIR}/backup_${TIMESTAMP}"

# Create backup directory
mkdir -p "${BACKUP_PATH}"

# Backup model files
echo "Backing up CNN models..."
tar -czf "${BACKUP_PATH}/models.tar.gz" model/

# Backup configuration
echo "Backing up configurations..."
cp -r configs/ "${BACKUP_PATH}/"
cp .env "${BACKUP_PATH}/"

# Backup recent jobs (last 30 days)
echo "Backing up recent jobs..."
find "${PIPELINE_JOBS_DIR}" -type d -name "job_*" -mtime -30 | \
    tar -czf "${BACKUP_PATH}/recent_jobs.tar.gz" -T -

# Backup logs
echo "Backing up logs..."
tar -czf "${BACKUP_PATH}/logs.tar.gz" logs/

# Create backup manifest
cat > "${BACKUP_PATH}/manifest.txt" << EOF
Backup created: ${TIMESTAMP}
Pipeline version: $(git rev-parse HEAD)
Models included: $(ls model/ | wc -l) learners
Jobs included: Recent jobs (30 days)
Size: $(du -sh "${BACKUP_PATH}" | cut -f1)
EOF

echo "Backup completed: ${BACKUP_PATH}"

# Cleanup old backups (keep last 7)
find "${BACKUP_DIR}" -type d -name "backup_*" | \
    sort -r | tail -n +8 | xargs rm -rf

echo "Cleanup completed"
```

### 3. Disaster Recovery

```python
# disaster_recovery.py
import os
import shutil
import json
from datetime import datetime

class DisasterRecoveryManager:
    def __init__(self, backup_location="/backup/frame_pipeline"):
        self.backup_location = backup_location
        
    def create_recovery_point(self):
        """Crea un punto di recovery completo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recovery_point = f"recovery_point_{timestamp}"
        
        recovery_data = {
            'timestamp': timestamp,
            'pipeline_state': self.capture_pipeline_state(),
            'job_states': self.capture_job_states(),
            'model_checksums': self.verify_model_integrity(),
            'config_snapshot': self.capture_config_snapshot()
        }
        
        recovery_file = os.path.join(
            self.backup_location, f"{recovery_point}.json"
        )
        
        with open(recovery_file, 'w') as f:
            json.dump(recovery_data, f, indent=2)
            
        return recovery_file
        
    def restore_from_recovery_point(self, recovery_point_file):
        """Ripristina da un punto di recovery"""
        
        with open(recovery_point_file, 'r') as f:
            recovery_data = json.load(f)
            
        print(f"Restoring from: {recovery_data['timestamp']}")
        
        # Restore models
        self.restore_models(recovery_data['model_checksums'])
        
        # Restore configuration
        self.restore_configuration(recovery_data['config_snapshot'])
        
        # Restore job states
        self.restore_job_states(recovery_data['job_states'])
        
        print("Recovery completed successfully")
        
    def verify_system_integrity(self):
        """Verifica integrità del sistema dopo recovery"""
        
        checks = {
            'models_available': self.check_models_available(),
            'config_valid': self.check_config_valid(),
            'dependencies_ok': self.check_dependencies(),
            'storage_accessible': self.check_storage_accessible()
        }
        
        all_checks_passed = all(checks.values())
        
        return {
            'integrity_ok': all_checks_passed,
            'checks': checks,
            'timestamp': datetime.now().isoformat()
        }
```

## Performance Tuning

### 1. Optimization Checklist

```markdown
## Performance Optimization Checklist

### Hardware Level
- [ ] GPU memory configuration optimized
- [ ] CPU core count matches workload
- [ ] SSD storage for temporary files
- [ ] Sufficient RAM (16GB+ recommended)
- [ ] Network bandwidth adequate for shared storage

### Software Level
- [ ] TensorFlow GPU support enabled
- [ ] OpenCV optimized build installed
- [ ] Python multiprocessing configured
- [ ] Batch sizes tuned for hardware
- [ ] Memory pooling enabled

### Configuration Level
- [ ] CNN input size optimized for GPU
- [ ] Ensemble size balanced with performance
- [ ] Preprocessing parameters tuned
- [ ] Output formats optimized
- [ ] Logging level appropriate

### System Level
- [ ] File system performance optimized
- [ ] Network storage configuration
- [ ] Process priority configured
- [ ] Resource limits set
- [ ] Monitoring enabled
```

### 2. Benchmarking Tools

```python
# benchmark.py
import time
import json
import numpy as np
from contextlib import contextmanager

class PipelineBenchmark:
    def __init__(self):
        self.results = {}
        
    @contextmanager
    def benchmark_section(self, section_name):
        """Context manager per benchmarking"""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            self.results[section_name] = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'timestamp': time.time()
            }
            
    def benchmark_full_pipeline(self, test_job_path):
        """Benchmark completo della pipeline"""
        
        with self.benchmark_section("total_pipeline"):
            # Simula esecuzione pipeline completa
            with self.benchmark_section("matlab_filter"):
                time.sleep(2)  # Simula processing MATLAB
                
            with self.benchmark_section("fix_groundtruth"):
                time.sleep(0.5)  # Simula correzione groundtruth
                
            with self.benchmark_section("fix_backgrounds"):
                time.sleep(1)  # Simula correzione background
                
            with self.benchmark_section("boost_filter"):
                time.sleep(5)  # Simula elaborazione CNN
                
            with self.benchmark_section("reconstruct"):
                time.sleep(1)  # Simula ricostruzione
                
        return self.results
        
    def generate_benchmark_report(self):
        """Genera report benchmark"""
        
        total_time = sum(r['duration'] for r in self.results.values())
        
        report = {
            'summary': {
                'total_duration': total_time,
                'total_memory_used': sum(r['memory_delta'] for r in self.results.values()),
                'timestamp': time.time()
            },
            'detailed_results': self.results,
            'performance_score': self.calculate_performance_score()
        }
        
        return report
```

---

**Riferimenti:**
- [Architecture Overview](01-architecture-overview.md)
- [Configuration Guide](02-configuration-guide.md)
- [Filtri e Algoritmi](03-filters-algorithms.md)
- [CNN Architecture](04-cnn-architecture.md)