# Frame Pipeline - Documentazione Tecnica

## Diagrammi e Visualizzazioni

> **Collezione Diagrammi**: Il sistema include diagrammi completi per visualizzare l'architettura:
> - **Cartella Sorgenti**: [`diagrams/`](diagrams/) - File PlantUML (.puml) editabili
> - **Cartella Immagini**: [`img/`](img/) - Diagrammi renderizzati (.png)
> - **Indice Completo**: [`DIAGRAMS_INDEX.md`](DIAGRAMS_INDEX.md) - Catalogo completo diagrammi
> - **Documentazione Diagrammi**: [`diagrams/README.md`](diagrams/README.md)

### Diagrammi Principali

| Diagramma | Sorgente PlantUML | Immagine | Descrizione |
|-----------|------------------|----------|-------------|
| **Architettura Sistema** | [high_level_architetture.puml](diagrams/high_level_architetture.puml) | ![Architecture](img/high_level_architetture.png) | Panoramica completa del sistema |
| **Flusso Dati** | [dataflow.puml](diagrams/dataflow.puml) | ![Dataflow](img/dataflow.png) | Pipeline step-by-step |
| **Catena Filtri** | [chain_architetture.puml](diagrams/chain_architetture.puml) | ![Filters](img/chain_architetture.png) | Architettura filtri |
| **Componenti** | [component_diagram.puml](diagrams/component_diagram.puml) | ![Components](img/component_diagram.png) | Diagramma componenti C4 |
| **Sequenza** | [sequence_diagram.puml](diagrams/sequence_diagram.puml) | ![Sequence](img/sequence_diagram.png) | Interazione temporale |

## Indice della Documentazione

Questa è la documentazione tecnica completa del Frame Pipeline

### Struttura Documentazione

| Documento | Descrizione | Contenuto Principale |
|-----------|-------------|---------------------|
| **[01-architecture-overview.md](01-architecture-overview.md)** | Panoramica architetturale del sistema | Architettura high-level, componenti principali, flusso dati |
| **[02-configuration-guide.md](02-configuration-guide.md)** | Guida completa alla configurazione | Parametri, profili, environment variables, best practices |
| **[03-filters-algorithms.md](03-filters-algorithms.md)** | Dettagli implementativi dei filtri | Algoritmi, preprocessing, CNN processing, ricostruzione |
| **[04-cnn-architecture.md](04-cnn-architecture.md)** | Architettura delle reti neurali | JigsawNet, ensemble learning, training, ottimizzazioni |
| **[05-deployment-guide.md](05-deployment-guide.md)** | Guide per deployment e produzione | Docker, cloud, monitoring, security, disaster recovery |
| **[DIAGRAMS_INDEX.md](DIAGRAMS_INDEX.md)** | Indice completo diagrammi | Catalogo diagrammi PlantUML e immagini con riferimenti |

### Pubblico di Riferimento

- **Sviluppatori**: Comprensione architettura e implementazione
- **DevOps/SysAdmin**: Deployment e gestione infrastruttura
- **Data Scientists**: Configurazione CNN e ottimizzazione algoritmi
- **Ricercatori**: Comprensione approcci algoritmici utilizzati

### Come Navigare la Documentazione

#### Per Nuovi Utenti
1. **Inizia con**: [Architecture Overview](01-architecture-overview.md) per comprendere il sistema
2. **Procedi con**: [Configuration Guide](02-configuration-guide.md) per la configurazione base
3. **Approfondisci**: [Filters & Algorithms](03-filters-algorithms.md) per i dettagli implementativi

#### Per Deployment
1. **Leggi**: [Deployment Guide](05-deployment-guide.md) per opzioni di deployment
2. **Consulta**: [Configuration Guide](02-configuration-guide.md) per ambienti specifici
3. **Riferimenti**: [Architecture Overview](01-architecture-overview.md) per requisiti sistema

#### Per Ricerca/Sviluppo
1. **Studio**: [CNN Architecture](04-cnn-architecture.md) per dettagli rete neurale
2. **Analisi**: [Filters & Algorithms](03-filters-algorithms.md) per algoritmi specifici
3. **Configurazione**: [Configuration Guide](02-configuration-guide.md) per tuning parametri

## Sistema Frame Pipeline

### Panoramica Rapida

Il Frame Pipeline è progettato per risolvere automaticamente puzzle di immagini frammentate attraverso una pipeline di processing in 6 fasi:

```
Input → MATLAB → Fix GT → Fix BG → CNN → Reconstruct → Output
```

1. **MATLAB Filter**: Preprocessing e generazione esempi training
2. **Fix Groundtruth**: Correzione dati di verità
3. **Fix Backgrounds**: Normalizzazione sfondi immagini
4. **Boost Filter**: Elaborazione CNN ensemble (5 reti)
5. **Reconstruct Filter**: Ricostruzione finale ottimizzata

### Workflow Visuale

**Pipeline Completo**: 
![Complete Pipeline Flow](img/dataflow_architetture.png)

**Trasformazioni Dati**: 
![Data Transformations](img/data_transformation.png)

### Caratteristiche Principali

- **CNN Ensemble**: 5 reti neurali specializzate per maggiore accuratezza
- **Automazione MATLAB**: Script PowerShell per esecuzione automatica step MATLAB
- **Job Management**: Sistema completo gestione job con ripresa automatica
- **Multi-Platform**: Supporto Linux/Windows con architetture ibride
- **Scalabilità**: Deploy locale, Docker, cloud (AWS/Azure/GCP)
- **Monitoring**: Sistema completo metriche, alerting e health checks

### Tecnologie Utilizzate

| Componente | Tecnologia | Versione | Scopo |
|------------|------------|----------|-------|
| **Pipeline Core** | Python | 3.7.9 | Orchestrazione e controllo |
| **CNN Framework** | TensorFlow | 1.15.x | Deep learning e inferenza |
| **Preprocessing** | MATLAB | R2020b+ | Generazione esempi training |
| **Computer Vision** | OpenCV | 4.11+ | Processing immagini |
| **Automation** | PowerShell | 5.0+ | Automazione Windows/MATLAB |
| **Containerization** | Docker | 20.10+ | Deployment e isolamento |
| **Orchestration** | Docker Compose | 1.29+ | Multi-container deployment |

### Requisiti di Sistema

#### Minimi
- **CPU**: 4 core, 2.5GHz+
- **RAM**: 8GB
- **Storage**: 20GB SSD
- **GPU**: Opzionale (CPU fallback)
- **OS**: Linux Ubuntu 18.04+ o Windows 10+

#### Consigliati
- **CPU**: 8+ core, 3.0GHz+
- **RAM**: 16GB+
- **Storage**: 50GB+ NVMe SSD
- **GPU**: NVIDIA GTX 1660+ (4GB+ VRAM)
- **Network**: 1Gbps per storage condiviso

## Quick Start

### Installazione Rapida

```bash
# 1. Clone repository
git clone <repository-url>
cd frame_pipeline

# 2. Setup ambiente
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configurazione
cp .env.example .env
# Edita .env con i tuoi percorsi

# 4. Test installazione
./run_pipeline.py --help
python3.7 test_boost.py all
```

### Primo Job

```bash
# Esegui pipeline su esempio
./run_pipeline.py \
  ./examples/puzzle_3x3/masks/ \
  ./examples/puzzle_3x3/original.jpg \
  "1:9"
```

### Architettura Sviluppo vs Produzione

> **Riferimento Visuale**: Vedi [System Architecture](img/high_level_architetture.png) per dettagli completi

#### Sviluppo (Singola Macchina)
```
┌─────────────────────────────────────┐
│        Development Machine          │
├─────────────────────────────────────┤
│  Python Pipeline + MATLAB           │
│  Local Storage                      │
│  Manual Job Management              │
└─────────────────────────────────────┘
```

#### Produzione (Distribuita)
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Linux Cluster │  │ Windows MATLAB  │  │ Shared Storage  │
│   (CNN + Logic) │  │  (Preprocessing)│  │ (Jobs + Models) │
│                 │  │                 │  │                 │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │ Container │  │  │  │PowerShell │  │  │  │    NFS    │  │
│  │ Instances │  │  │  │ Monitor   │  │  │  │    S3     │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Guida per Tipologie di Utente

### Per Data Scientists

**Focus**: Ottimizzazione algoritmi CNN e tuning hyperparameters

**Documenti Chiave**:
- [CNN Architecture](04-cnn-architecture.md) - Architettura rete neurale
- [Configuration Guide](02-configuration-guide.md) - Tuning parametri
- [Filters & Algorithms](03-filters-algorithms.md) - Algoritmi preprocessing

**Parametri Critici**:
```json
{
  "hyperparameters": {
    "width": 160,           // Dimensione input CNN
    "height": 160,
    "batch_size": 64,       // Tuning per GPU
    "learning_rate": 1e-4,  // Convergenza training
    "learner_num": 5        // Dimensione ensemble
  }
}
```

### Per DevOps Engineers

**Focus**: Deployment, scaling, monitoring

**Documenti Chiave**:
- [Deployment Guide](05-deployment-guide.md) - Strategie deployment
- [Architecture Overview](01-architecture-overview.md) - Requisiti infrastruttura
- [Configuration Guide](02-configuration-guide.md) - Environment configuration

**Configurazioni Critiche**:
```yaml
# Docker Compose per produzione
services:
  pipeline:
    image: frame-pipeline:latest
    replicas: 3
    resources:
      limits:
        memory: 4G
        cpus: '2'
```

### Per Software Developers

**Focus**: Estensioni, customizzazioni, debugging

**Documenti Chiave**:
- [Filters & Algorithms](03-filters-algorithms.md) - Implementazione filtri
- [Architecture Overview](01-architecture-overview.md) - Struttura codice
- [Configuration Guide](02-configuration-guide.md) - Debug configuration

**Pattern di Estensione**:
```python
# Nuovo filtro custom
class MyCustomFilter(BaseFilter):
    def process(self):
        # Implementazione logica specifica
        return True
```

### Per System Administrators

**Focus**: Installazione, manutenzione, backup

**Documenti Chiave**:
- [Deployment Guide](05-deployment-guide.md) - Setup sistemi
- [Configuration Guide](02-configuration-guide.md) - Configurazione ambiente
- [Architecture Overview](01-architecture-overview.md) - Requisiti sistema

**Script di Manutenzione**:
```bash
# Health check automatico
./scripts/health_check.sh

# Backup automatico
./scripts/backup_pipeline.sh
```

## Risoluzione Problemi Comuni

### FAQ Rapide

| Problema | Causa Comune | Soluzione Rapida |
|----------|--------------|------------------|
| Import errors | Ambiente Python | `pip install -r requirements.txt` |
| MATLAB non trovato | Path mancante | Aggiorna `MATLAB_PATH` in .env |
| GPU non rilevata | Driver CUDA | Verifica `nvidia-smi` |
| Out of memory | Batch size alto | Riduci `batch_size` in config |
| Jobs non processati | PowerShell policy | `Set-ExecutionPolicy RemoteSigned` |

### Log Locations

| Componente | Location | Descrizione |
|------------|----------|-------------|
| Pipeline Python | `job_folder/logs/pipeline.log` | Log principale pipeline |
| MATLAB Steps | `job_folder/logs/matlab_execution_*.log` | Log esecuzione MATLAB |
| CNN Processing | `job_folder/logs/boost_filter.log` | Log elaborazione CNN |
| System Health | `/var/log/frame_pipeline/` | Log sistema generale |

## Contribuire alla Documentazione

### Struttura Documenti

Ogni documento segue questa struttura:

1. **Introduzione**: Overview e scope
2. **Sezioni Tecniche**: Dettagli implementativi
3. **Esempi Pratici**: Code snippets e configurazioni
4. **Riferimenti**: Link ad altri documenti

### Come Aggiornare

```bash
# 1. Modifica documento
vim docs/XX-document-name.md

# 2. Valida markdown
markdownlint docs/

# 3. Test links
markdown-link-check docs/*.md

# 4. Commit changes
git add docs/
git commit -m "Update documentation: [description]"
```

### Guidelines

- **Linguaggio**: Tecnico ma accessibile
- **Esempi**: Sempre includere code examples
- **Links**: Cross-reference tra documenti
- **Versioning**: Aggiorna con modifiche al codice

---

## Informazioni di Versione

| Campo | Valore |
|-------|--------|
| **Versione Documentazione** | 1.0.0 |
| **Versione Pipeline** | 1.0.0 |
| **Ultima Modifica** | November 2025 |
| **Autori** | Frame Pipeline Team |
| **Licenza** | MIT |

**Nota**: Questa documentazione è viva e viene aggiornata regolarmente. Per la versione più recente, consulta il repository Git.
