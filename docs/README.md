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
| **[04-deployment-guide.md](04-deployment-guide.md)** | Guide per deployment e produzione | Docker, cloud, monitoring, security, disaster recovery |
| **[DIAGRAMS_INDEX.md](DIAGRAMS_INDEX.md)** | Indice completo diagrammi | Catalogo diagrammi PlantUML e immagini con riferimenti |
| **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** | Installazione sistema | Manuale d'installazione di tutto il sistema della FrameAI Pipline |

### Pubblico di Riferimento

- **Sviluppatori**: Comprensione architettura e implementazione
- **DevOps/SysAdmin**: Deployment e gestione infrastruttura
- **Data Scientists**: Configurazione CNN e ottimizzazione algoritmi
- **Ricercatori**: Comprensione approcci algoritmici utilizzati

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

## Risoluzione Problemi Comuni

### FAQ Rapide

| Problema | Causa Comune | Soluzione Rapida |
|----------|--------------|------------------|
| Import errors | Ambiente Python | `pip install -r requirements.txt` |
| MATLAB non trovato | Path mancante | Aggiorna `MATLAB_PATH` in .env |
| GPU non rilevata | Driver CUDA | Verifica `nvidia-smi` |
| Out of memory | Batch size alto | Riduci `batch_size` in config |
| Jobs non processati | PowerShell policy | `Set-ExecutionPolicy RemoteSigned` |
