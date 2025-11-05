# Frame Pipeline - Filtri e Algoritmi

> **Diagrammi di Riferimento**: 
> - [Filter Chain Architecture](img/chain_architetture.png)
> - [Data Transformation Flow](img/data_transformation.png)
> - [Sequence Diagram](img/sequence_diagram.png)
> - [PlantUML Sources](diagrams/)

## Introduzione

Questo documento fornisce una descrizione dettagliata di ogni filtro nella pipeline, inclusi gli algoritmi utilizzati, i parametri di configurazione e le ottimizzazioni implementate.

## Architettura dei Filtri

> **Diagramma Architettura**: ![Filter Chain](img/chain_architetture.png)

### Base Filter Framework

Tutti i filtri ereditano da una classe base comune che fornisce l'interfaccia standard:

```python
class BaseFilter:
    def __init__(self, shared_dir, mask_folder, original_image, 
                 config_file, parts, job_folder, job_id):
        self.shared_dir = shared_dir
        self.mask_folder = mask_folder
        self.original_image = original_image
        self.config_file = config_file
        self.parts = parts
        self.job_folder = job_folder
        self.job_id = job_id
        self.parameters = None
        
    def load_parameters(self):
        """Carica parametri da Parameters singleton"""
        pass
        
    def process(self):
        """Implementazione specifica del filtro"""
        raise NotImplementedError
        
    def get_output_dir(self):
        """Directory output specifica del filtro"""
        pass
```

## Filtro 1: MATLAB Filter

> **Diagrammi Specifici**: 
> - [Pipeline Flow](img/dataflow.png) - Mostra MATLAB Filter come Step 1-2
> - [Sequence Diagram](img/sequence_diagram.png) - Interazione MATLAB Runtime

### Scopo e Funzionalità

Il MATLAB Filter è responsabile del preprocessing iniziale dei frammenti del puzzle e della generazione dei dati di training per la CNN.

![MATLAB Processing Flow](img/dataflow.png)

### Algoritmi Implementati

#### Step 1: Generate New Example

**Input:**
- Directory con frammenti (maschere)
- Immagine originale di riferimento
- Parametri di configurazione

**Processo:**
1. **Caricamento Frammenti:**
   ```matlab
   fragments = load_fragments(masks_folder, mask_extension);
   original = imread(original_image_path);
   ```

2. **Preprocessing Immagini:**
   ```matlab
   for each fragment:
       % Normalizzazione dimensioni
       fragment_resized = imresize(fragment, [target_height, target_width]);
       
       % Rimozione background
       mask = create_background_mask(fragment, bg_color, bg_tolerance);
       fragment_clean = apply_mask(fragment_resized, mask);
       
       % Normalizzazione colori
       fragment_normalized = normalize_colors(fragment_clean);
   ```

3. **Generazione Esempi Training:**
   ```matlab
   % Genera combinazioni di frammenti per training
   training_pairs = generate_fragment_pairs(fragments);
   
   for each pair:
       % Calcola ground truth alignment
       [translation, rotation] = compute_alignment(pair.frag1, pair.frag2);
       
       % Salva esempio training
       save_training_example(pair, alignment, output_dir);
   ```

**Output:**
- Directory `m/` con esempi di training
- Metadati di allineamento
- Statistiche preprocessing

#### Step 2: Export Alignments

**Input:**
- Esempi generati dallo Step 1
- Configurazione algoritmi

**Processo:**
1. **Analisi Geometrica:**
   ```matlab
   for each training_example:
       % Estrazione features geometriche
       edges = extract_edges(example.fragment);
       corners = detect_corners(edges);
       
       % Calcolo descrittori forma
       shape_descriptors = compute_shape_descriptors(edges, corners);
   ```

2. **Calcolo Allineamenti:**
   ```matlab
   % Algoritmo di matching basato su features
   alignments = [];
   for i = 1:num_fragments:
       for j = i+1:num_fragments:
           alignment = compute_pairwise_alignment(fragments(i), fragments(j));
           if alignment.confidence > threshold:
               alignments = [alignments; alignment];
           end
       end
   ```

3. **Esportazione Risultati:**
   ```matlab
   % Formato output: [frag1_id, frag2_id, tx, ty, rotation, confidence]
   save_alignments(alignments, output_file);
   ```

### Automazione PowerShell

#### Monitoraggio e Esecuzione

Il sistema PowerShell monitora automaticamente nuovi job e esegue i passaggi MATLAB:

```powershell
function Execute-MatlabJobs($jobFolder) {
    # Caricamento parametri
    $params = Get-Content "$jobFolder/input/params.json" | ConvertFrom-Json
    
    # Step 1: Generate New Example
    $result1 = Invoke-MatlabScript -ScriptName "generate_new_example" `
        -Parameters @($masksFolder, $originalImage, $outputDir, $masksExt, $outputExt)
    
    # Step 2: Export Alignments  
    if ($result1.Success) {
        $result2 = Invoke-MatlabScript -ScriptName "export_alignments" `
            -Parameters @($step1OutputDir, $alignmentsFile, $configFile)
    }
}
```

#### Gestione Errori e Logging

```powershell
# Timeout management
$TimeoutMinutes = 30

# Structured logging
$logEntry = @{
    Timestamp = Get-Date
    JobFolder = $jobFolder
    Step = "generate_new_example"
    Status = "Running"
    Parameters = $parameters
}

# Error recovery
try {
    $result = & matlab -batch $matlabScript
} catch {
    Write-Log "MATLAB execution failed: $($_.Exception.Message)"
    # Retry logic o escalation
}
```

## Filtro 2: Fix Groundtruth Filter

### Scopo e Funzionalità

Corregge e valida i dati di ground truth generati da MATLAB, rimuovendo allineamenti errati e normalizzando i formati.

### Algoritmi di Correzione

#### 1. Validazione Allineamenti

```python
def validate_alignments(alignments_file):
    """Valida la consistenza degli allineamenti"""
    
    alignments = load_alignments(alignments_file)
    valid_alignments = []
    
    for alignment in alignments:
        # Controllo validità geometrica
        if is_geometrically_valid(alignment):
            # Controllo consistenza con neighbors
            if is_consistent_with_neighbors(alignment, valid_alignments):
                valid_alignments.append(alignment)
            else:
                log_warning(f"Inconsistent alignment: {alignment}")
        else:
            log_warning(f"Invalid geometry: {alignment}")
    
    return valid_alignments
```

#### 2. Correzione Errori Sistematici

```python
def correct_systematic_errors(alignments):
    """Corregge errori sistematici negli allineamenti"""
    
    # Identificazione bias sistematici
    translation_bias = compute_translation_bias(alignments)
    rotation_bias = compute_rotation_bias(alignments)
    
    corrected_alignments = []
    for alignment in alignments:
        # Correzione bias
        alignment.tx -= translation_bias.x
        alignment.ty -= translation_bias.y
        alignment.rotation -= rotation_bias
        
        # Normalizzazione range
        alignment.rotation = normalize_angle(alignment.rotation)
        
        corrected_alignments.append(alignment)
    
    return corrected_alignments
```

#### 3. Filtering e Smoothing

```python
def apply_smoothing_filters(alignments):
    """Applica filtri di smoothing agli allineamenti"""
    
    # Filtro mediano per rimozione outliers
    filtered_alignments = median_filter(alignments, window_size=3)
    
    # Smoothing gaussiano per ridurre noise
    smoothed_alignments = gaussian_filter(filtered_alignments, sigma=1.0)
    
    # Interpolazione per allineamenti mancanti
    complete_alignments = interpolate_missing(smoothed_alignments)
    
    return complete_alignments
```

### Formati e Conversioni

#### Input Format (MATLAB)
```
# Fragment_A Fragment_B Translation_X Translation_Y Rotation Confidence
1 2 10.5 -5.2 15.3 0.85
2 3 -8.1 12.7 -22.1 0.92
...
```

#### Output Format (Standardized)
```json
{
  "alignments": [
    {
      "fragment_a": 1,
      "fragment_b": 2,
      "translation": {"x": 10.5, "y": -5.2},
      "rotation": 15.3,
      "confidence": 0.85,
      "validated": true
    }
  ],
  "metadata": {
    "total_alignments": 156,
    "valid_alignments": 142,
    "correction_applied": true
  }
}
```

## Filtro 3: Fix Image Backgrounds Filter

### Scopo e Funzionalità

Normalizza i background delle immagini dei frammenti per garantire consistenza nell'elaborazione CNN.

### Algoritmi di Background Processing

#### 1. Background Detection

```python
def detect_background_color(image, tolerance=0):
    """Rileva automaticamente il colore di background predominante"""
    
    # Campionamento bordi immagine
    border_pixels = extract_border_pixels(image, border_width=5)
    
    # Clustering colori per identificare background
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit(border_pixels)
    
    # Selezione cluster più frequente come background
    bg_cluster = select_most_frequent_cluster(clusters)
    bg_color = bg_cluster.center
    
    return bg_color, calculate_tolerance(bg_cluster)
```

#### 2. Background Removal

```python
def remove_background(image, bg_color, tolerance):
    """Rimuove il background dall'immagine"""
    
    # Conversione in spazio colore LAB per migliore separazione
    lab_image = rgb2lab(image)
    lab_bg_color = rgb2lab(bg_color)
    
    # Calcolo distanza euclidea per ogni pixel
    distances = np.sqrt(np.sum((lab_image - lab_bg_color) ** 2, axis=2))
    
    # Creazione maschera background
    background_mask = distances <= tolerance
    
    # Rimozione background
    result = image.copy()
    result[background_mask] = [255, 255, 255]  # Bianco standard
    
    return result, background_mask
```

#### 3. Edge Preservation

```python
def preserve_fragment_edges(image, mask):
    """Preserva i bordi del frammento durante la rimozione background"""
    
    # Rilevamento bordi con Canny
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    
    # Dilatazione bordi per preservare dettagli
    dilated_edges = cv2.dilate(edges, kernel=np.ones((3,3)), iterations=1)
    
    # Modifica maschera per preservare bordi
    preserved_mask = mask & ~dilated_edges.astype(bool)
    
    return preserved_mask
```

#### 4. Color Normalization

```python
def normalize_fragment_colors(image):
    """Normalizza i colori del frammento"""
    
    # Normalizzazione istogramma
    normalized = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    normalized_rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    
    # Correzione gamma per migliorare contrasto
    gamma_corrected = adjust_gamma(normalized_rgb, gamma=1.2)
    
    # Normalizzazione range [0,1]
    final_normalized = gamma_corrected.astype(np.float32) / 255.0
    
    return final_normalized
```

## Filtro 4: Boost Filter (CNN Processing)

### Scopo e Funzionalità

Questo è il filtro principale che utilizza le reti neurali convoluzionali per predire gli allineamenti ottimali tra i frammenti.

### Architettura CNN

#### JigsawNet Architecture

```python
class JigsawNetWithROI:
    def __init__(self):
        self.width = 160
        self.height = 160
        self.depth = 3
        self.learner_num = 5  # Ensemble di 5 reti
        
    def build_network(self):
        """Costruisce l'architettura della rete"""
        
        # Input layer
        input_layer = tf.placeholder(tf.float32, 
                                   [None, self.height, self.width, self.depth])
        
        # Convolutional layers
        conv1 = self.conv_layer(input_layer, 32, 'conv1')
        pool1 = self.max_pool(conv1)
        
        conv2 = self.conv_layer(pool1, 64, 'conv2') 
        pool2 = self.max_pool(conv2)
        
        conv3 = self.conv_layer(pool2, 128, 'conv3')
        pool3 = self.max_pool(conv3)
        
        # Fully connected layers
        flattened = tf.layers.flatten(pool3)
        fc1 = tf.layers.dense(flattened, 512, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(fc1, rate=0.5)
        
        fc2 = tf.layers.dense(dropout1, 256, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(fc2, rate=0.5)
        
        # Output layer: [tx, ty, rotation, confidence]
        output = tf.layers.dense(dropout2, 4, activation=None)
        
        return input_layer, output
```

#### Ensemble Learning

```python
def ensemble_prediction(self, fragment_pair):
    """Combina predizioni di multiple reti"""
    
    predictions = []
    confidences = []
    
    # Predizioni da ogni learner
    for learner_id in range(self.learner_num):
        model_path = f"model/g{learner_id}/"
        prediction = self.single_prediction(fragment_pair, model_path)
        
        predictions.append(prediction[:3])  # tx, ty, rotation
        confidences.append(prediction[3])   # confidence
    
    # Weighted averaging basato su confidence
    weights = np.array(confidences) / np.sum(confidences)
    final_prediction = np.average(predictions, axis=0, weights=weights)
    final_confidence = np.mean(confidences)
    
    return np.append(final_prediction, final_confidence)
```

### Preprocessing per CNN

#### 1. Data Augmentation

```python
def augment_training_data(image_pairs):
    """Applica data augmentation agli esempi di training"""
    
    augmented_pairs = []
    
    for pair in image_pairs:
        # Rotazioni multiple
        for angle in [0, 90, 180, 270]:
            rotated_pair = rotate_image_pair(pair, angle)
            augmented_pairs.append(rotated_pair)
        
        # Flip orizzontale/verticale
        h_flipped = flip_horizontal(pair)
        v_flipped = flip_vertical(pair)
        augmented_pairs.extend([h_flipped, v_flipped])
        
        # Variazioni luminosità
        for brightness in [0.8, 1.0, 1.2]:
            bright_pair = adjust_brightness(pair, brightness)
            augmented_pairs.append(bright_pair)
    
    return augmented_pairs
```

#### 2. Feature Normalization

```python
def normalize_features(image):
    """Normalizzazione features per CNN"""
    
    # Resize alla dimensione target
    resized = cv2.resize(image, (self.width, self.height))
    
    # Normalizzazione [0,1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Zero-center e unit variance
    mean = np.mean(normalized, axis=(0,1), keepdims=True)
    std = np.std(normalized, axis=(0,1), keepdims=True) + 1e-8
    standardized = (normalized - mean) / std
    
    return standardized
```

### Ottimizzazioni Performance

#### 1. Batch Processing

```python
def process_fragments_batch(self, fragments, batch_size=64):
    """Elabora frammenti in batch per efficienza"""
    
    num_fragments = len(fragments)
    results = []
    
    for i in range(0, num_fragments, batch_size):
        batch = fragments[i:i+batch_size]
        
        # Preprocessing batch
        batch_preprocessed = [self.preprocess_fragment(f) for f in batch]
        batch_tensor = np.stack(batch_preprocessed)
        
        # Inferenza batch
        batch_predictions = self.session.run(
            self.output_tensor,
            feed_dict={self.input_tensor: batch_tensor}
        )
        
        results.extend(batch_predictions)
    
    return results
```

#### 2. Memory Management

```python
def optimize_memory_usage(self):
    """Ottimizza l'uso della memoria durante l'inferenza"""
    
    # Configurazione TensorFlow per memoria limitata
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    # Cleanup esplicito tensori temporanei
    tf.reset_default_graph()
    
    # Garbage collection periodico
    if self.processed_fragments % 100 == 0:
        gc.collect()
```

## Filtro 5: Reconstruct Filter

### Scopo e Funzionalità

Ricostruisce l'immagine finale utilizzando le predizioni della CNN e un algoritmo di ottimizzazione globale.

### Algoritmo GlobalReassembly

#### 1. Input Processing

```python
def prepare_reconstruction_data(self, boost_output_dir):
    """Prepara i dati per la ricostruzione"""
    
    # Caricamento predizioni CNN
    predictions_file = os.path.join(boost_output_dir, "predictions.json")
    predictions = self.load_predictions(predictions_file)
    
    # Caricamento frammenti originali
    fragments = self.load_fragments(self.fragments_dir)
    
    # Creazione dataset_list.txt per GlobalReassembly
    dataset_list = self.create_dataset_list(predictions, fragments)
    
    return dataset_list
```

#### 2. Global Optimization

Il binario GlobalReassembly implementa un algoritmo di ottimizzazione globale:

```cpp
// Pseudocodice dell'algoritmo C++
class GlobalReassembly {
public:
    ReconstructionResult optimize(const DatasetList& fragments) {
        // Inizializzazione griglia puzzle
        PuzzleGrid grid = initializeGrid(fragments.size());
        
        // Algoritmo simulated annealing
        double temperature = initial_temperature;
        Solution current_solution = generateInitialSolution(fragments);
        Solution best_solution = current_solution;
        
        while (temperature > min_temperature) {
            // Genera vicino casuale
            Solution neighbor = generateNeighbor(current_solution);
            
            // Calcola delta energia
            double delta_energy = computeEnergy(neighbor) - computeEnergy(current_solution);
            
            // Accettazione/rigetto
            if (delta_energy < 0 || acceptanceProbability(delta_energy, temperature) > random()) {
                current_solution = neighbor;
                
                if (computeEnergy(current_solution) < computeEnergy(best_solution)) {
                    best_solution = current_solution;
                }
            }
            
            temperature *= cooling_rate;
        }
        
        return generateFinalImage(best_solution);
    }
};
```

#### 3. Energy Function

```python
def compute_alignment_energy(self, fragment_positions):
    """Calcola l'energia di allineamento per la configurazione corrente"""
    
    total_energy = 0.0
    
    for i, pos_i in enumerate(fragment_positions):
        for j, pos_j in enumerate(fragment_positions):
            if i >= j:
                continue
                
            # Distanza fra frammenti
            distance = np.linalg.norm(pos_i - pos_j)
            
            # Predizione CNN per questa coppia
            predicted_alignment = self.get_prediction(i, j)
            
            # Energia basata su discrepanza predizione vs posizione
            alignment_error = self.compute_alignment_error(
                pos_i, pos_j, predicted_alignment
            )
            
            # Peso basato su confidence CNN
            weight = predicted_alignment.confidence
            
            total_energy += weight * alignment_error
    
    return total_energy
```

#### 4. Final Assembly

```python
def assemble_final_image(self, optimized_positions, fragments):
    """Assembla l'immagine finale dalle posizioni ottimizzate"""
    
    # Calcola dimensioni canvas finale
    canvas_width, canvas_height = self.compute_canvas_size(optimized_positions)
    
    # Crea canvas vuoto
    final_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Posiziona ogni frammento
    for fragment, position in zip(fragments, optimized_positions):
        x, y, rotation = position
        
        # Applica rotazione al frammento
        rotated_fragment = self.rotate_fragment(fragment, rotation)
        
        # Blend con canvas esistente
        final_image = self.blend_fragment(
            final_image, rotated_fragment, x, y
        )
    
    return final_image
```

### Gestione Output

#### Dataset List Format

Il file `dataset_list.txt` contiene le informazioni necessarie per la ricostruzione:

```
# Fragment_ID X_Position Y_Position Rotation Confidence
1 145.2 67.8 15.3 0.92
2 298.1 45.2 -8.7 0.89
3 67.5 156.9 22.1 0.95
...
```

#### Risultati Finali

```python
def save_reconstruction_results(self, final_image, metadata):
    """Salva i risultati della ricostruzione"""
    
    # Immagine finale
    cv2.imwrite(
        os.path.join(self.output_dir, "reconstructed_puzzle.jpg"),
        final_image
    )
    
    # Metadati ricostruzione
    result_metadata = {
        "reconstruction_time": metadata.processing_time,
        "final_energy": metadata.final_energy,
        "iterations": metadata.iterations,
        "fragments_used": len(metadata.fragments),
        "success_rate": metadata.placement_accuracy
    }
    
    with open(os.path.join(self.output_dir, "reconstruction_results.json"), 'w') as f:
        json.dump(result_metadata, f, indent=2)
```

## Performance e Ottimizzazioni

### Parallelizzazione

```python
from multiprocessing import Pool

def parallel_filter_processing(self, fragments):
    """Elaborazione parallela dei filtri dove possibile"""
    
    with Pool(processes=os.cpu_count()) as pool:
        # Background processing parallelo
        background_results = pool.map(self.process_background, fragments)
        
        # CNN inference parallela (se multiple GPU)
        if self.gpu_count > 1:
            cnn_results = pool.map(self.cnn_inference, background_results)
        else:
            cnn_results = [self.cnn_inference(f) for f in background_results]
    
    return cnn_results
```

### Memory Optimization

```python
def optimize_memory_footprint(self):
    """Ottimizza l'uso della memoria durante l'elaborazione"""
    
    # Lazy loading dei dati
    fragment_generator = self.lazy_load_fragments()
    
    # Processing in chunks
    chunk_size = self.calculate_optimal_chunk_size()
    
    for chunk in self.chunk_generator(fragment_generator, chunk_size):
        results = self.process_chunk(chunk)
        self.save_intermediate_results(results)
        
        # Cleanup memoria chunk corrente
        del chunk, results
        gc.collect()
```

---

**Riferimenti:**
- [Architecture Overview](01-architecture-overview.md)
- [Configuration Guide](02-configuration-guide.md)
- [CNN Architecture](04-cnn-architecture.md)