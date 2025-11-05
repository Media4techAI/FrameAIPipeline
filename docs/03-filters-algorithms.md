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

Converte in formato python i dati di ground truth generati da MATLAB.

## Filtro 3: Fix Image Backgrounds Filter

### Scopo e Funzionalità

Normalizza i background delle immagini dei frammenti per garantire consistenza nell'elaborazione CNN.

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

## Filtro 5: Reconstruct Filter

Ricostruisce l'immagine finale utilizzando le predizioni della CNN e un algoritmo di ricostruzione finale implementato in C++.

**Riferimenti:**
- [Architecture Overview](01-architecture-overview.md)
- [Configuration Guide](02-configuration-guide.md)
- [Deployment Guide](04-deployment-guide.md)