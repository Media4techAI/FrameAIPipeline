# Frame Pipeline - Configurazione Dettagliata

> **Diagrammi di Riferimento**: 
> - [High Level Architecture](img/high_level_architetture.png)
> - [Component Diagram](img/component_diagram.png)
> - [Configuration PlantUML](diagrams/high_level_architetture.puml)

## Introduzione

Questo documento fornisce una guida completa alla configurazione del Frame Pipeline, dalla configurazione base agli scenari avanzati e personalizzazioni.

## Sistema di Configurazione

> **Diagramma Configurazione**: Vedere l'architettura gerarchica nella [Panoramica Architetturale](01-architecture-overview.md#configuration-system)

### Gerarchia delle Configurazioni

Il sistema utilizza una struttura gerarchica a cascata per le configurazioni:

```
1. Default Hardcoded Values (lowest priority)
2. Template Configuration (config_template.json)
3. User Configuration File (custom.json)
4. Job-Specific Configuration (job/input/params.json)
5. Environment Variables (.env)
6. Runtime Arguments (highest priority)
```

### File di Configurazione Principale

#### Struttura Base `params.json`

```json
{
  "job_id": "job_20231103_150644",
  "input_dir": "input",
  "masks_dir": "input/masks",
  "original_image": "input/original.jpg",
  "parts": "1:9",
  
  "output_steps_dirs": {
    "step1": "output/step1_generate_new_example",
    "step2": "output/step2_export_alignments", 
    "step3": "output/step3_fix_groundtruth",
    "step4": "output/step4_fix_images",
    "step5": "output/step5_boost",
    "step6": "output/step6_reconstruct"
  },
  
  "bg_color": "255 255 255",
  "bg_tolerance": 0,
  "mask_extension": ".jpg",
  "output_extension": ".jpg",
  "top_k": 10,
  
  "load_options": { ... },
  "dis_options": { ... },
  "confoptions": { ... },
  "hyperparameters": { ... }
}
```

## Sezioni di Configurazione Dettagliate

### 1. Parametri di Base

#### Job Identification
```json
{
  "job_id": "string",           // ID univoco del job
  "input_dir": "string",        // Directory input relativa
  "masks_dir": "string",        // Directory frammenti
  "original_image": "string",   // Path immagine originale
  "parts": "string"             // Range parti (es: "1:9", "1:20")
}
```

**Validazione:**
- `job_id`: Deve essere univoco, formato consigliato: `job_YYYYMMDD_HHMMSS`
- `parts`: Formato "inizio:fine", entrambi numeri interi positivi
- Paths: Devono essere relativi alla root del job

#### Output Configuration
```json
{
  "output_steps_dirs": {
    "step1": "output/step1_generate_new_example",
    "step2": "output/step2_export_alignments",
    "step3": "output/step3_fix_groundtruth", 
    "step4": "output/step4_fix_images",
    "step5": "output/step5_boost",
    "step6": "output/step6_reconstruct"
  }
}
```

**Note:**
- Tutti i path sono relativi alla directory del job
- Le directory vengono create automaticamente
- Modificare solo se necessario per debugging

### 2. Image Processing Options

#### Background Processing
```json
{
  "bg_color": "255 255 255",    // RGB del background
  "bg_tolerance": 0,            // Tolleranza colore (0-255)
  "mask_extension": ".jpg",     // Estensione file frammenti
  "output_extension": ".jpg"    // Estensione output
}
```

**Configurazioni Comuni:**
```json
// Background bianco standard
{
  "bg_color": "255 255 255",
  "bg_tolerance": 0
}

// Background nero
{
  "bg_color": "0 0 0", 
  "bg_tolerance": 5
}

// Background verde (chroma key)
{
  "bg_color": "0 255 0",
  "bg_tolerance": 20
}

// Background automatico (tolleranza alta)
{
  "bg_color": "128 128 128",
  "bg_tolerance": 50
}
```

### 3. Load Options (Preprocessing)

#### Configurazione Completa
```json
{
  "load_options": {
    "is_grad_dir_on_plt_output": false,     // Mostra direzione gradiente
    "is_smooth_before_grad": false,         // Smooth pre-gradiente
    "is_use_rand_frag_rot": false,          // Rotazione casuale frammenti
    "is_use_dist_colors_4grad": true,       // Colori distinti gradiente
    "reduce_poly_length_to": 0.2,           // Riduzione lunghezza poligoni
    "palette_sz": 14,                       // Dimensione palette colori
    "is_use_palette_pxls_vals": true        // Usa valori palette
  }
}
```

#### Profili Predefiniti

**Alta Qualità (Slow & Precise):**
```json
{
  "load_options": {
    "is_smooth_before_grad": true,
    "reduce_poly_length_to": 0.1,
    "palette_sz": 20,
    "is_use_dist_colors_4grad": true
  }
}
```

**Performance (Fast & Approximated):**
```json
{
  "load_options": {
    "is_smooth_before_grad": false,
    "reduce_poly_length_to": 0.3,
    "palette_sz": 8,
    "is_use_rand_frag_rot": true
  }
}
```

### 4. Dissimilarity Options

#### Configurazione Matching
```json
{
  "dis_options": {
    "sampling_res": 4,              // Risoluzione campionamento
    "small_ov_percentage": 0.05,    // % overlap piccolo
    "smoothing_deg_window": 7       // Finestra smoothing (gradi)
  }
}
```

**Tuning Guidelines:**
- `sampling_res`: Più alto = più preciso ma più lento (1-16)
- `small_ov_percentage`: Soglia overlap minimo (0.01-0.1)
- `smoothing_deg_window`: Finestra smoothing angoli (1-30)

### 5. Confidence Options (Matching)

#### Parametri Confidence
```json
{
  "confoptions": {
    "iou_thresh": 0.7,      // Soglia IoU per matching
    "top_i": 1,             // Top-i candidati
    "gamma_H": 0.7,         // Gamma alto
    "gamma_L": 0.3          // Gamma basso
  }
}
```

**Tuning per Diversi Scenari:**

**Puzzle Semplici (pochi pezzi, bordi chiari):**
```json
{
  "confoptions": {
    "iou_thresh": 0.6,
    "top_i": 1,
    "gamma_H": 0.8,
    "gamma_L": 0.2
  }
}
```

**Puzzle Complessi (molti pezzi, texture simili):**
```json
{
  "confoptions": {
    "iou_thresh": 0.8,
    "top_i": 3,
    "gamma_H": 0.6,
    "gamma_L": 0.4
  }
}
```

### 6. CNN Hyperparameters

#### Configurazione Rete Neurale
```json
{
  "hyperparameters": {
    "width": 160,                    // Larghezza input CNN
    "height": 160,                   // Altezza input CNN  
    "depth": 3,                      // Canali (RGB=3)
    "batch_size": 64,                // Dimensione batch
    "weight_decay": 1e-4,            // Decadimento pesi (L2)
    "learning_rate": 1e-4,           // Learning rate
    "total_training_step": 30000,    // Step training totali
    "learner_num": 5                 // Numero learner ensemble
  }
}
```

#### Profili Hardware-Specific

**GPU Potente (RTX 3080+):**
```json
{
  "hyperparameters": {
    "width": 224,
    "height": 224,
    "batch_size": 32,
    "learner_num": 7,
    "total_training_step": 50000
  }
}
```

**GPU Media (GTX 1660, RTX 2060):**
```json
{
  "hyperparameters": {
    "width": 160,
    "height": 160, 
    "batch_size": 16,
    "learner_num": 5,
    "total_training_step": 30000
  }
}
```

**CPU Only:**
```json
{
  "hyperparameters": {
    "width": 128,
    "height": 128,
    "batch_size": 8,
    "learner_num": 3,
    "total_training_step": 15000
  }
}
```

## Configurazioni Environment (.env)

### File .env Base
```env
# ===========================================
# FRAME PIPELINE CONFIGURATION
# ===========================================

# Job Management
PIPELINE_JOBS_DIR=/mnt/c/Users/user/Documents/Reassembly2d_Sources/jobs
PIPELINE_DEFAULT_CONFIG=/home/frame/frame_pipeline/config_template.json
```

### Environment Variables Reference

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `PIPELINE_JOBS_DIR` | Directory principale job | `./jobs` | `/path/to/jobs` |
| `PIPELINE_DEFAULT_CONFIG` | Config predefinito | None | `/path/to/config.json` |

**Riferimenti:**
- [Architecture Overview](01-architecture-overview.md)
- [Filtri e Algoritmi](03-filters-algorithms.md)
- [Deployment Guide](04-deployment-guide.md)