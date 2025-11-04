# Frame Pipeline - CNN Architecture

## Introduzione

Questo documento fornisce una descrizione dettagliata dell'architettura delle reti neurali convoluzionali utilizzate nel Frame Pipeline, incluse le specifiche tecniche, le ottimizzazioni implementate e le strategie di training.

## Architettura JigsawNet

### Overview dell'Architettura

Il sistema utilizza un'architettura CNN specializzata chiamata **JigsawNetWithROI** progettata specificamente per il problema dell'allineamento di frammenti di puzzle. L'architettura è ottimizzata per elaborare coppie di frammenti e predire le loro relazioni spaziali.

```
Input Pair (160x160x3 each)
         ↓
    Preprocessing
         ↓
┌─────────────────────┐
│   Feature Extractor │
│                     │
│  Conv1 (32 filters) │
│  Pool1 (2x2)        │
│  Conv2 (64 filters) │
│  Pool2 (2x2)        │
│  Conv3 (128 filters)│
│  Pool3 (2x2)        │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Feature Fusion     │
│                     │
│  Concatenate        │
│  FC1 (512 units)    │
│  Dropout (0.5)      │
│  FC2 (256 units)    │
│  Dropout (0.5)      │
└─────────────────────┘
         ↓
┌─────────────────────┐
│   Regression Head   │
│                     │
│  FC3 (4 units)      │
│  [tx, ty, θ, conf]  │
└─────────────────────┘
```

### Implementazione Dettagliata

#### 1. Input Processing Layer

```python
class InputProcessor:
    def __init__(self, width=160, height=160, depth=3):
        self.width = width
        self.height = height
        self.depth = depth
        
    def preprocess_fragment_pair(self, fragment1, fragment2):
        """Preprocessa una coppia di frammenti per la CNN"""
        
        # Resize alle dimensioni target
        frag1_resized = cv2.resize(fragment1, (self.width, self.height))
        frag2_resized = cv2.resize(fragment2, (self.width, self.height))
        
        # Normalizzazione [0,1]
        frag1_norm = frag1_resized.astype(np.float32) / 255.0
        frag2_norm = frag2_resized.astype(np.float32) / 255.0
        
        # Concatenazione canali (160x160x6)
        combined_input = np.concatenate([frag1_norm, frag2_norm], axis=2)
        
        # Data augmentation (se training)
        if self.training_mode:
            combined_input = self.apply_augmentation(combined_input)
            
        return combined_input
        
    def apply_augmentation(self, input_tensor):
        """Applica data augmentation durante il training"""
        
        # Rotazione casuale
        angle = np.random.uniform(-15, 15)
        rotated = self.rotate_tensor(input_tensor, angle)
        
        # Flip casuale
        if np.random.random() > 0.5:
            rotated = np.fliplr(rotated)
            
        # Variazione luminosità
        brightness_factor = np.random.uniform(0.8, 1.2)
        augmented = np.clip(rotated * brightness_factor, 0, 1)
        
        # Noise gaussiano
        noise = np.random.normal(0, 0.01, augmented.shape)
        final = np.clip(augmented + noise, 0, 1)
        
        return final
```

#### 2. Convolutional Feature Extractor

```python
class ConvolutionalExtractor:
    def __init__(self):
        self.layers = []
        
    def build_extractor(self, input_tensor):
        """Costruisce le layers convoluzionali"""
        
        # Conv Block 1
        conv1 = tf.layers.conv2d(
            inputs=input_tensor,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            name="conv1"
        )
        
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2,
            name="pool1"
        )
        
        # Batch Normalization
        norm1 = tf.layers.batch_normalization(
            inputs=pool1,
            training=self.is_training,
            name="norm1"
        )
        
        # Conv Block 2
        conv2 = tf.layers.conv2d(
            inputs=norm1,
            filters=64,
            kernel_size=[5, 5],
            padding="same", 
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            name="conv2"
        )
        
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2,
            name="pool2"
        )
        
        norm2 = tf.layers.batch_normalization(
            inputs=pool2,
            training=self.is_training,
            name="norm2"
        )
        
        # Conv Block 3  
        conv3 = tf.layers.conv2d(
            inputs=norm2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            name="conv3"
        )
        
        pool3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=[2, 2], 
            strides=2,
            name="pool3"
        )
        
        norm3 = tf.layers.batch_normalization(
            inputs=pool3,
            training=self.is_training,
            name="norm3"
        )
        
        return norm3
```

#### 3. Feature Fusion e Dense Layers

```python
class FeatureFusion:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        
    def build_fusion_layers(self, conv_features):
        """Costruisce le layers di fusione delle features"""
        
        # Flatten delle features convoluzionali
        flattened = tf.layers.flatten(conv_features, name="flatten")
        
        # Dense Layer 1
        fc1 = tf.layers.dense(
            inputs=flattened,
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name="fc1"
        )
        
        # Dropout per regolarizzazione
        dropout1 = tf.layers.dropout(
            inputs=fc1,
            rate=self.dropout_rate,
            training=self.is_training,
            name="dropout1"
        )
        
        # Dense Layer 2
        fc2 = tf.layers.dense(
            inputs=dropout1,
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name="fc2"
        )
        
        dropout2 = tf.layers.dropout(
            inputs=fc2,
            rate=self.dropout_rate,
            training=self.is_training,
            name="dropout2"
        )
        
        return dropout2
```

#### 4. Regression Head

```python
class RegressionHead:
    def __init__(self):
        self.output_size = 4  # [tx, ty, rotation, confidence]
        
    def build_regression_head(self, features):
        """Costruisce la testa di regressione per predire allineamenti"""
        
        # Output layer lineare
        predictions = tf.layers.dense(
            inputs=features,
            units=self.output_size,
            activation=None,  # Linear activation
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.zeros_initializer(),
            name="predictions"
        )
        
        # Split delle predizioni
        translation_x = predictions[:, 0]
        translation_y = predictions[:, 1] 
        rotation = predictions[:, 2]
        confidence_logit = predictions[:, 3]
        
        # Normalizzazione outputs
        # Translation: range [-1, 1] normalizzato rispetto alle dimensioni immagine
        tx_normalized = tf.tanh(translation_x) * self.max_translation
        ty_normalized = tf.tanh(translation_y) * self.max_translation
        
        # Rotation: range [-π, π]
        rotation_normalized = tf.tanh(rotation) * np.pi
        
        # Confidence: range [0, 1]
        confidence = tf.sigmoid(confidence_logit)
        
        final_output = tf.stack([
            tx_normalized, 
            ty_normalized, 
            rotation_normalized, 
            confidence
        ], axis=1)
        
        return final_output
```

### Ensemble Architecture

#### Multi-Learner System

Il sistema utilizza un ensemble di 5 reti neurali identiche (g0, g1, g2, g3, g4) addestrate su subset diversi dei dati per migliorare robustezza e accuratezza.

```python
class EnsembleJigsawNet:
    def __init__(self, num_learners=5):
        self.num_learners = num_learners
        self.learners = []
        self.model_paths = [f"model/g{i}/" for i in range(num_learners)]
        
    def load_ensemble(self):
        """Carica tutti i modelli dell'ensemble"""
        
        for i, model_path in enumerate(self.model_paths):
            print(f"Loading learner {i} from {model_path}")
            
            # Carica checkpoint del modello
            learner = self.load_single_learner(model_path)
            self.learners.append(learner)
            
    def ensemble_predict(self, fragment_pair):
        """Combina predizioni di tutti i learners"""
        
        predictions = []
        confidences = []
        
        # Ottieni predizione da ogni learner
        for learner in self.learners:
            pred = learner.predict(fragment_pair)
            
            tx, ty, rotation, confidence = pred
            predictions.append([tx, ty, rotation])
            confidences.append(confidence)
        
        # Weighted averaging basato su confidence
        weights = np.array(confidences)
        weights = weights / np.sum(weights)  # Normalizza
        
        # Media pesata delle predizioni
        final_prediction = np.average(predictions, axis=0, weights=weights)
        final_confidence = np.mean(confidences)
        
        # Gestione outliers
        final_prediction = self.remove_outlier_predictions(
            predictions, final_prediction, threshold=2.0
        )
        
        return np.append(final_prediction, final_confidence)
        
    def remove_outlier_predictions(self, predictions, mean_pred, threshold):
        """Rimuove predizioni outlier dall'ensemble"""
        
        valid_predictions = []
        
        for pred in predictions:
            distance = np.linalg.norm(np.array(pred) - mean_pred)
            if distance <= threshold:
                valid_predictions.append(pred)
                
        if len(valid_predictions) > 0:
            return np.mean(valid_predictions, axis=0)
        else:
            return mean_pred  # Fallback su media originale
```

## Training Strategy

### 1. Dataset Preparation

```python
class TrainingDataGenerator:
    def __init__(self, fragments_dir, original_image):
        self.fragments_dir = fragments_dir
        self.original_image = original_image
        
    def generate_training_pairs(self):
        """Genera coppie di training con ground truth"""
        
        fragments = self.load_fragments()
        training_pairs = []
        
        for i, frag1 in enumerate(fragments):
            for j, frag2 in enumerate(fragments):
                if i >= j:
                    continue
                    
                # Calcola ground truth alignment
                gt_alignment = self.compute_ground_truth_alignment(frag1, frag2)
                
                # Crea positive e negative examples
                positive_pair = {
                    'fragment1': frag1,
                    'fragment2': frag2,
                    'target': gt_alignment,
                    'label': 1.0  # Positive example
                }
                
                # Negative examples con misalignment casuale
                negative_pairs = self.generate_negative_examples(frag1, frag2)
                
                training_pairs.append(positive_pair)
                training_pairs.extend(negative_pairs)
                
        return training_pairs
        
    def compute_ground_truth_alignment(self, frag1, frag2):
        """Calcola l'allineamento ground truth tra due frammenti"""
        
        # Estrazione features SIFT
        kp1, desc1 = self.sift.detectAndCompute(frag1, None)
        kp2, desc2 = self.sift.detectAndCompute(frag2, None)
        
        # Matching features
        matches = self.matcher.match(desc1, desc2)
        
        # RANSAC per stima trasformazione robusta
        if len(matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            transformation, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC
            )
            
            # Estrazione parametri trasformazione
            tx, ty = transformation[0, 2], transformation[1, 2]
            rotation = np.arctan2(transformation[1, 0], transformation[0, 0])
            
            return [tx, ty, rotation]
        else:
            return [0, 0, 0]  # No reliable alignment found
```

### 2. Loss Function Design

```python
class AlignmentLoss:
    def __init__(self, translation_weight=1.0, rotation_weight=0.5, 
                 confidence_weight=0.3):
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight  
        self.confidence_weight = confidence_weight
        
    def compute_loss(self, predictions, targets, is_positive):
        """Calcola la loss per allineamento e confidence"""
        
        pred_tx, pred_ty, pred_rot, pred_conf = tf.split(predictions, 4, axis=1)
        target_tx, target_ty, target_rot = tf.split(targets, 3, axis=1)
        
        # Translation loss (L2)
        translation_loss = tf.reduce_mean(
            tf.square(pred_tx - target_tx) + tf.square(pred_ty - target_ty)
        )
        
        # Rotation loss (angular difference)
        rotation_diff = self.angular_difference(pred_rot, target_rot)
        rotation_loss = tf.reduce_mean(tf.square(rotation_diff))
        
        # Confidence loss (binary cross-entropy)
        confidence_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=is_positive, logits=pred_conf
            )
        )
        
        # Combined loss
        total_loss = (
            self.translation_weight * translation_loss +
            self.rotation_weight * rotation_loss +
            self.confidence_weight * confidence_loss
        )
        
        return total_loss, {
            'translation_loss': translation_loss,
            'rotation_loss': rotation_loss,
            'confidence_loss': confidence_loss
        }
        
    def angular_difference(self, angle1, angle2):
        """Calcola la differenza angolare normalizzata"""
        diff = angle1 - angle2
        # Normalizza in [-π, π]
        normalized_diff = tf.atan2(tf.sin(diff), tf.cos(diff))
        return normalized_diff
```

### 3. Training Loop

```python
class TrainingLoop:
    def __init__(self, model, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        
    def train(self, training_data, validation_data, num_epochs=100):
        """Loop di training principale"""
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch(training_data)
            
            # Validation phase
            val_loss = self.validate_epoch(validation_data)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    def train_epoch(self, training_data):
        """Training per una epoch"""
        
        total_loss = 0
        num_batches = 0
        
        for batch in training_data:
            fragment_pairs, targets, labels = batch
            
            with tf.GradientTape() as tape:
                predictions = self.model(fragment_pairs, training=True)
                loss, loss_components = self.loss_function.compute_loss(
                    predictions, targets, labels
                )
                
            # Backpropagation
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
            
            total_loss += loss
            num_batches += 1
            
        return total_loss / num_batches
```

## Ottimizzazioni e Performance

### 1. Memory Optimization

```python
class MemoryOptimizer:
    def __init__(self):
        self.memory_growth = True
        self.memory_limit = None
        
    def configure_gpu_memory(self):
        """Configura l'uso della memoria GPU"""
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Abilita memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                # Imposta limite memoria se specificato
                if self.memory_limit:
                    tf.config.experimental.set_memory_limit(
                        gpus[0], self.memory_limit
                    )
                    
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                
    def optimize_inference_memory(self, model):
        """Ottimizza memoria durante l'inferenza"""
        
        # Congela grafo per inferenza
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            self.session, self.graph.as_graph_def(), ["output"]
        )
        
        # Ottimizzazioni del grafo
        optimized_graph = tf.graph_util.optimize_for_inference(
            frozen_graph, ["input"], ["output"], 
            tf.uint8.as_datatype_enum
        )
        
        return optimized_graph
```

### 2. Batch Processing Optimization

```python
class BatchProcessor:
    def __init__(self, batch_size=64, prefetch_buffer=2):
        self.batch_size = batch_size
        self.prefetch_buffer = prefetch_buffer
        
    def create_optimized_dataset(self, data_generator):
        """Crea dataset ottimizzato con prefetching"""
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape([None, 160, 160, 6]),  # Fragment pairs
                tf.TensorShape([None, 3]),            # Targets
                tf.TensorShape([None, 1])             # Labels
            )
        )
        
        # Ottimizzazioni pipeline
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer)
        dataset = dataset.cache()  # Cache in memoria se possibile
        
        return dataset
```

### 3. Model Quantization

```python
class ModelQuantizer:
    def __init__(self):
        self.quantization_mode = "dynamic"
        
    def quantize_model(self, model_path, output_path):
        """Quantizza il modello per inference più veloce"""
        
        # Carica modello
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        
        # Configurazione quantizzazione
        if self.quantization_mode == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif self.quantization_mode == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            
        # Conversione
        quantized_model = converter.convert()
        
        # Salvataggio
        with open(output_path, 'wb') as f:
            f.write(quantized_model)
            
        print(f"Quantized model saved to {output_path}")
```

## Model Evaluation e Metrics

### 1. Alignment Accuracy Metrics

```python
class AlignmentMetrics:
    def __init__(self, translation_threshold=5.0, rotation_threshold=0.1):
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        
    def compute_alignment_accuracy(self, predictions, ground_truth):
        """Calcola accuratezza allineamento"""
        
        pred_tx, pred_ty, pred_rot, pred_conf = predictions.T
        gt_tx, gt_ty, gt_rot = ground_truth.T
        
        # Translation accuracy
        translation_error = np.sqrt(
            (pred_tx - gt_tx)**2 + (pred_ty - gt_ty)**2
        )
        translation_accuracy = np.mean(
            translation_error <= self.translation_threshold
        )
        
        # Rotation accuracy  
        rotation_error = np.abs(self.angular_difference(pred_rot, gt_rot))
        rotation_accuracy = np.mean(
            rotation_error <= self.rotation_threshold
        )
        
        # Combined accuracy
        combined_accuracy = np.mean(
            (translation_error <= self.translation_threshold) &
            (rotation_error <= self.rotation_threshold)
        )
        
        return {
            'translation_accuracy': translation_accuracy,
            'rotation_accuracy': rotation_accuracy,
            'combined_accuracy': combined_accuracy,
            'mean_translation_error': np.mean(translation_error),
            'mean_rotation_error': np.mean(rotation_error)
        }
```

### 2. Confidence Calibration

```python
class ConfidenceCalibrator:
    def __init__(self):
        self.calibration_bins = 10
        
    def calibrate_confidence(self, predictions, accuracies):
        """Calibra le confidence predictions"""
        
        confidences = predictions[:, 3]  # Confidence scores
        
        # Binning per calibrazione
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_curve = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Trova predizioni in questo bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                
                calibration_curve.append({
                    'bin_range': (bin_lower, bin_upper),
                    'avg_confidence': avg_confidence,
                    'avg_accuracy': avg_accuracy,
                    'count': in_bin.sum()
                })
                
        return calibration_curve
```

---

**Riferimenti:**
- [Architecture Overview](01-architecture-overview.md)
- [Configuration Guide](02-configuration-guide.md)
- [Filtri e Algoritmi](03-filters-algorithms.md)
- [Deployment Guide](05-deployment-guide.md)