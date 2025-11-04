# Documentazione Config Template

Questo documento spiega tutti i parametri di configurazione disponibili nel file `config_template.json`.

## Parametri di Base

### `job_id`
- **Tipo**: string
- **Descrizione**: Identificatore univoco del job (viene sostituito automaticamente dallo script)
- **Valore template**: `"PLACEHOLDER_JOB_ID"`

### `input_dir`, `masks_dir`, `original_image`
- **Tipo**: string
- **Descrizione**: Percorsi relativi alle directory e file di input
- **Default**: 
  - `input_dir`: `"input"`
  - `masks_dir`: `"input/masks"`
  - `original_image`: `"input/original.jpg"`

### `output_steps_dirs`
- **Tipo**: object
- **Descrizione**: Directory di output per ogni step della pipeline
- **Steps**:
  - `step1`: Generazione frammenti (`"output/step1_generate_new_example"`)
  - `step2`: Esportazione allineamenti (`"output/step2_export_alignments"`)
  - `step3`: Correzione ground truth (`"output/step3_fix_groundtruth"`)
  - `step4`: Correzione immagini (`"output/step4_fix_images"`)
  - `step5`: Boost processing (`"output/step5_boost"`)
  - `step6`: Ricostruzione finale (`"output/step6_reconstruct"`)

## Configurazione di Base

### `bg_color`
- **Tipo**: string
- **Descrizione**: Colore di sfondo in formato "R G B"
- **Esempio**: `"255 255 255"` (bianco), `"8 248 8"` (verde chiaro)
- **Valore template**: `"PLACEHOLDER_BG_COLOR"`

### `bg_tolerance`
- **Tipo**: integer
- **Descrizione**: Tolleranza per il riconoscimento del colore di sfondo (distanza euclidea)
- **Default**: `0`
- **Range**: 0-255
- **Note**: Valori più alti rendono il riconoscimento del background più permissivo

### `mask_extension`
- **Tipo**: string
- **Descrizione**: Estensione dei file maschera
- **Default**: `".jpg"`
- **Valori possibili**: `".jpg"`, `".png"`, `".bmp"`, etc.

### `output_extension`
- **Tipo**: string
- **Descrizione**: Estensione dei file di output
- **Default**: `".jpg"`
- **Valori possibili**: `".jpg"`, `".png"`, `".bmp"`, etc.

### `parts`
- **Tipo**: string
- **Descrizione**: Range dei frammenti da processare
- **Formato**: `"inizio:fine"` (es: `"1:9"`)
- **Valore template**: `"PLACEHOLDER_PARTS"`

### `top_k`
- **Tipo**: integer
- **Descrizione**: Numero di migliori candidati da considerare nell'algoritmo
- **Default**: `10`
- **Range**: 1-100

## Load Options (Opzioni di Caricamento)

### `is_grad_dir_on_plt_output`
- **Tipo**: boolean
- **Descrizione**: Abilita la visualizzazione della direzione del gradiente nell'output del plot
- **Default**: `false`

### `is_smooth_before_grad`
- **Tipo**: boolean
- **Descrizione**: Applica smoothing prima del calcolo del gradiente
- **Default**: `false`

### `is_use_rand_frag_rot`
- **Tipo**: boolean
- **Descrizione**: Applica rotazione casuale ai frammenti durante il preprocessing
- **Default**: `false`

### `is_use_dist_colors_4grad`
- **Tipo**: boolean
- **Descrizione**: Usa colori distinti per il calcolo del gradiente
- **Default**: `true`

### `reduce_poly_length_to`
- **Tipo**: float
- **Descrizione**: Riduce la lunghezza dei poligoni al valore specificato (frazione)
- **Default**: `0.2`
- **Range**: 0.0-1.0

### `palette_sz`
- **Tipo**: integer
- **Descrizione**: Dimensione della palette dei colori
- **Default**: `14`
- **Range**: 1-256

### `is_use_palette_pxls_vals`
- **Tipo**: boolean
- **Descrizione**: Usa i valori dei pixel dalla palette invece di colori calcolati
- **Default**: `true`

## Dis Options (Opzioni di Dissimilarità)

### `sampling_res`
- **Tipo**: integer
- **Descrizione**: Risoluzione di campionamento per il calcolo della dissimilarità
- **Default**: `4`
- **Range**: 1-16

### `small_ov_percentage`
- **Tipo**: float
- **Descrizione**: Percentuale minima di overlap considerata "piccola"
- **Default**: `0.05`
- **Range**: 0.0-1.0

### `smoothing_deg_window`
- **Tipo**: integer
- **Descrizione**: Dimensione della finestra di smoothing in gradi
- **Default**: `7`
- **Range**: 1-360

## Altri Parametri

### `dis_version`
- **Tipo**: integer
- **Descrizione**: Versione dell'algoritmo di calcolo della dissimilarità
- **Default**: `2`
- **Valori possibili**: `1`, `2`

### `n_reduced_tforms`
- **Tipo**: integer
- **Descrizione**: Numero massimo di trasformazioni ridotte da considerare
- **Default**: `40000`
- **Range**: 1000-100000

## Conf Options (Opzioni di Configurazione Matching)

### `iou_thresh`
- **Tipo**: float
- **Descrizione**: Soglia Intersection over Union per il matching
- **Default**: `0.7`
- **Range**: 0.0-1.0

### `top_i`
- **Tipo**: integer
- **Descrizione**: Numero di migliori match da considerare per ogni frammento
- **Default**: `1`
- **Range**: 1-10

### `gamma_H`
- **Tipo**: float
- **Descrizione**: Parametro gamma alto per il weighting nel matching
- **Default**: `0.7`
- **Range**: 0.0-1.0

### `gamma_L`
- **Tipo**: float
- **Descrizione**: Parametro gamma basso per il weighting nel matching
- **Default**: `0.3`
- **Range**: 0.0-1.0

## Hyperparameters (CNN Training Parameters)

### `width`
- **Tipo**: integer
- **Descrizione**: Larghezza delle immagini di input per la CNN
- **Default**: `160`
- **Range**: 32-512
- **Note**: Deve essere compatibile con l'architettura della rete

### `height`
- **Tipo**: integer
- **Descrizione**: Altezza delle immagini di input per la CNN
- **Default**: `160`
- **Range**: 32-512
- **Note**: Deve essere compatibile con l'architettura della rete

### `depth`
- **Tipo**: integer
- **Descrizione**: Numero di canali delle immagini di input (RGB = 3)
- **Default**: `3`
- **Valori possibili**: `1` (grayscale), `3` (RGB), `4` (RGBA)

### `batch_size`
- **Tipo**: integer
- **Descrizione**: Dimensione del batch per il training della CNN
- **Default**: `64`
- **Range**: 1-256
- **Note**: Valori più alti richiedono più memoria GPU

### `weight_decay`
- **Tipo**: float
- **Descrizione**: Coefficiente di decadimento dei pesi (regularizzazione L2)
- **Default**: `1e-4`
- **Format**: Notazione scientifica (es: `1e-4` = 0.0001)
- **Range**: 1e-6 - 1e-2

### `learning_rate`
- **Tipo**: float
- **Descrizione**: Tasso di apprendimento per l'ottimizzatore
- **Default**: `1e-4`
- **Format**: Notazione scientifica (es: `1e-4` = 0.0001)
- **Range**: 1e-6 - 1e-1

### `total_training_step`
- **Tipo**: integer
- **Descrizione**: Numero totale di step di training
- **Default**: `30000`
- **Range**: 1000-100000
- **Note**: Più step = training più lungo ma potenzialmente migliori risultati

### `learner_num`
- **Tipo**: integer
- **Descrizione**: Numero di learner nel modello ensemble
- **Default**: `5`
- **Range**: 1-10
- **Note**: Più learner = maggiore accuratezza ma training più lento

## Come Usare il Template

1. **Copia il template**:
   ```bash
   cp config_template.json my_config.json
   ```

2. **Modifica i parametri** nel file `my_config.json` secondo le tue esigenze

3. **Usa la configurazione personalizzata**:
   ```bash
   ./run_pipeline.sh ./masks/ ./orig.jpg "1:9" "255 255 255" my_config.json
   ```

## Esempi di Configurazioni

### Configurazione per Alta Precisione
```json
{
  "top_k": 20,
  "load_options": {
    "is_smooth_before_grad": true,
    "reduce_poly_length_to": 0.1
  },
  "confoptions": {
    "iou_thresh": 0.8,
    "top_i": 3
  }
}
```

### Configurazione per Performance
```json
{
  "top_k": 5,
  "n_reduced_tforms": 20000,
  "dis_options": {
    "sampling_res": 2
  }
}
```

### Configurazione CNN per Training Veloce
```json
{
  "hyperparameters": {
    "width": 128,
    "height": 128,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "total_training_step": 15000,
    "learner_num": 3
  }
}
```

### Configurazione CNN per Alta Qualità
```json
{
  "hyperparameters": {
    "width": 224,
    "height": 224,
    "batch_size": 16,
    "weight_decay": 5e-5,
    "learning_rate": 5e-5,
    "total_training_step": 50000,
    "learner_num": 7
  }
}
```