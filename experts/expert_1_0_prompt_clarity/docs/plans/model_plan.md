# Expert 1.0 Prompt Clarity Model Plan

## Overview
The Prompt Clarity Model is designed to evaluate and classify the clarity of prompts, helping to ensure high-quality inputs for AI systems. This model will be implemented as a binary classifier using DistilBERT architecture, providing a more efficient solution while maintaining strong performance.

## Model Architecture
1. Base Model Specifications
   - Model: DistilBERT-base-uncased
   - Vocabulary Size: 30,522 tokens
   - Hidden Size: 768
   - Number of Layers: 6 (distilled from 12)
   - Number of Attention Heads: 12
   - Maximum Sequence Length: 512 tokens
   - Position Embeddings: Learned, max position 512
   - Token Type Embeddings: 2 types
   - Model Size: ~260MB (compared to ~440MB for BERT-base)

2. Classification Head
   - Input: [CLS] token representation (768 dimensions)
   - Hidden Layer: Linear(768, 256) + ReLU
   - Dropout: 0.1
   - Output Layer: Linear(256, 2) + Softmax
   - Output: Binary classification probabilities

3. Model Configuration
   ```python
   {
       "model_type": "distilbert",
       "hidden_size": 768,
       "num_hidden_layers": 6,
       "num_attention_heads": 12,
       "intermediate_size": 3072,
       "hidden_dropout_prob": 0.1,
       "attention_probs_dropout_prob": 0.1,
       "max_position_embeddings": 512,
       "type_vocab_size": 2,
       "initializer_range": 0.02
   }
   ```

## Data Pipeline
1. Data Collection Specifications
   - Source 1: Original High Clarity Prompts
     * Format: CSV
     * Files:
       - clarity_high_short.csv (51 samples)
       - clarity_high_medium.csv (205 samples)
       - clarity_high_long.csv (44 samples)
     * Total: 300 samples
     * Categories: Short, Medium, Long prompts
   
   - Source 2: Original Low Clarity Prompts
     * Format: CSV
     * Files:
       - clarity_low_short.csv (195 samples)
       - clarity_low_medium.csv (119 samples)
       - clarity_low_long.csv (65 samples)
     * Total: 379 samples
     * Categories: Short, Medium, Long prompts
   
   - Source 3: Back-translated Augmented Data
     * Format: CSV
     * Files:
       - clarity_high_short_back_translated.csv (52 samples)
       - clarity_high_medium_back_translated.csv (206 samples)
       - clarity_high_long_back_translated.csv (45 samples)
       - clarity_low_short_back_translated.csv (196 samples)
       - clarity_low_medium_back_translated.csv (120 samples)
       - clarity_low_long_back_translated.csv (66 samples)
     * Total: 685 samples
     * Categories: Short, Medium, Long prompts
   
   - Source 4: Pre-split Datasets
     * Format: CSV
     * Files:
       - clarity_train_set.csv (952 samples)
       - clarity_test_set.csv (240 samples)
     * Total: 1,192 samples
     * Purpose: Ready-to-use train/test split

   - Source 5: Merged Dataset
     * Format: CSV
     * File: clarity_merged.csv (2,536 samples)
     * Purpose: Complete dataset for custom splits

2. Data Preprocessing Pipeline
   ```python
   def preprocess_text(text):
       # 1. Basic cleaning
       text = text.lower()
       text = re.sub(r'[^\w\s]', '', text)
       
       # 2. Tokenization
       tokens = tokenizer(
           text,
           padding='max_length',
           truncation=True,
           max_length=512,
           return_tensors='pt'
       )
       
       # 3. Special token handling
       tokens['input_ids'] = tokens['input_ids'].squeeze()
       tokens['attention_mask'] = tokens['attention_mask'].squeeze()
       
       return tokens
   ```

3. Data Augmentation Strategy
   - Back-translation: English → French → English
   - Synonym replacement: 20% of words
   - Random insertion: 10% of sentences
   - Random deletion: 5% of words

4. Train/Test Split
   - Option 1: Use pre-split datasets
     * Training: 952 samples (clarity_train_set.csv)
     * Test: 240 samples (clarity_test_set.csv)
   
   - Option 2: Custom split from merged dataset
     * Training: 80% (stratified by clarity score)
     * Validation: 10%
     * Test: 10%
     * Cross-validation: 5 folds

## Fine-Tuning Strategy
1. Dataset Imbalance Analysis
   - High Clarity: 300 samples
   - Low Clarity: 379 samples
   - Back-translated High: 303 samples
   - Back-translated Low: 382 samples
   - Total Imbalance: ~1.26:1 (Low:High)

2. Balancing Techniques
   ```python
   def balance_dataset(dataset, strategy='combined'):
       if strategy == 'oversampling':
           # Oversample minority class
           high_clarity = dataset[dataset['clarity'] == 1]
           low_clarity = dataset[dataset['clarity'] == 0]
           
           # Oversample high clarity to match low clarity
           high_clarity_oversampled = resample(
               high_clarity,
               replace=True,
               n_samples=len(low_clarity),
               random_state=42
           )
           
           return pd.concat([high_clarity_oversampled, low_clarity])
           
       elif strategy == 'undersampling':
           # Undersample majority class
           high_clarity = dataset[dataset['clarity'] == 1]
           low_clarity = dataset[dataset['clarity'] == 0]
           
           # Undersample low clarity to match high clarity
           low_clarity_undersampled = resample(
               low_clarity,
               replace=False,
               n_samples=len(high_clarity),
               random_state=42
           )
           
           return pd.concat([high_clarity, low_clarity_undersampled])
           
       elif strategy == 'combined':
           # Use both original and back-translated data
           # This naturally balances the dataset while maintaining diversity
           return dataset
   ```

3. Fine-Tuning Process
   ```python
   def fine_tune_model(model, train_dataset, val_dataset):
       # 1. Initial training with class weights
       class_weights = compute_class_weights(train_dataset)
       
       # 2. First phase: Train on balanced dataset
       trainer = Trainer(
           model=model,
           args=training_args,
           train_dataset=balance_dataset(train_dataset, 'combined'),
           eval_dataset=val_dataset,
           compute_metrics=compute_metrics
       )
       trainer.train()
       
       # 3. Second phase: Fine-tune on original distribution
       trainer = Trainer(
           model=model,
           args=training_args,
           train_dataset=train_dataset,  # Original distribution
           eval_dataset=val_dataset,
           compute_metrics=compute_metrics
       )
       trainer.train()
       
       return model
   ```

4. Class Weighting
   ```python
   def compute_class_weights(dataset):
       # Compute class weights based on inverse frequency
       class_counts = dataset['clarity'].value_counts()
       total_samples = len(dataset)
       
       weights = {
           0: total_samples / (2 * class_counts[0]),  # Low clarity
           1: total_samples / (2 * class_counts[1])   # High clarity
       }
       
       return weights
   ```

5. Evaluation Strategy
   - Use stratified k-fold cross-validation
   - Monitor both overall accuracy and per-class metrics
   - Track confusion matrix for each fold
   - Use F1-score as primary metric

6. Fine-Tuning Hyperparameters
   ```python
   fine_tuning_args = {
       "learning_rate": 1e-5,  # Lower learning rate for fine-tuning
       "per_device_train_batch_size": 16,
       "num_train_epochs": 2,
       "weight_decay": 0.01,
       "warmup_ratio": 0.1,
       "evaluation_strategy": "steps",
       "eval_steps": 100,
       "save_strategy": "steps",
       "save_steps": 100,
       "load_best_model_at_end": True,
       "metric_for_best_model": "f1",
       "greater_is_better": True
   }
   ```

7. Monitoring and Adjustment
   - Track per-class accuracy during training
   - Monitor validation loss for both classes
   - Adjust class weights if needed
   - Implement early stopping if overfitting occurs

## Training Process
1. Model Configuration
   ```python
   training_args = {
       "learning_rate": 3e-5,  # Slightly higher learning rate for DistilBERT
       "per_device_train_batch_size": 32,  # Increased batch size due to smaller model
       "per_device_eval_batch_size": 32,
       "num_train_epochs": 3,
       "weight_decay": 0.01,
       "evaluation_strategy": "epoch",
       "save_strategy": "epoch",
       "load_best_model_at_end": True,
       "metric_for_best_model": "f1",
       "greater_is_better": True,
       "fp16": True,
       "gradient_accumulation_steps": 2,  # Reduced due to larger batch size
       "warmup_steps": 500,
       "logging_steps": 100
   }
   ```

2. Training Pipeline
   ```python
   def train_pipeline():
       # 1. Data loading
       train_dataset = load_and_preprocess("train")
       val_dataset = load_and_preprocess("val")
       
       # 2. Model initialization
       model = AutoModelForSequenceClassification.from_pretrained(
           "distilbert-base-uncased",
           num_labels=2
       )
       
       # 3. Training setup
       trainer = Trainer(
           model=model,
           args=training_args,
           train_dataset=train_dataset,
           eval_dataset=val_dataset,
           compute_metrics=compute_metrics
       )
       
       # 4. Training execution
       trainer.train()
       
       # 5. Model saving
       trainer.save_model("./models/final")
   ```

3. Checkpointing Strategy
   - Save frequency: Every epoch
   - Keep last: 3 checkpoints
   - Save format: PyTorch state dict
   - Checkpoint contents:
     * Model weights
     * Optimizer state
     * Training arguments
     * Tokenizer configuration

## Evaluation Metrics
1. Primary Metrics Implementation
   ```python
   def compute_metrics(eval_pred):
       predictions, labels = eval_pred
       predictions = np.argmax(predictions, axis=1)
       
       return {
           'accuracy': accuracy_score(labels, predictions),
           'f1': f1_score(labels, predictions),
           'precision': precision_score(labels, predictions),
           'recall': recall_score(labels, predictions)
       }
   ```

2. Secondary Metrics
   - Confusion Matrix: sklearn.metrics.confusion_matrix
   - ROC Curve: sklearn.metrics.roc_curve
   - AUC Score: sklearn.metrics.auc

## Model Deployment
1. Docker Container Specifications
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY src/ ./src/
   COPY models/ ./models/
   
   ENV PYTHONPATH=/app
   ENV MODEL_PATH=/app/models/final
   
   CMD ["python", "src/api.py"]
   ```

2. API Endpoints Implementation
   ```python
   from fastapi import FastAPI
   
   app = FastAPI()
   
   @app.post("/predict")
   async def predict(prompt: str):
       # Implementation
   
   @app.post("/batch_predict")
   async def batch_predict(prompts: List[str]):
       # Implementation
   
   @app.get("/health")
   async def health_check():
       # Implementation
   ```

## Monitoring and Maintenance
1. Performance Monitoring
   - Metrics collection: Prometheus
   - Visualization: Grafana
   - Logging: ELK Stack
   - Alerting: AlertManager

2. Model Updates
   - Version control: Git LFS
   - Model registry: MLflow
   - Experiment tracking: Weights & Biases
   - CI/CD: GitHub Actions

## Success Criteria
1. Model Performance
   - Accuracy: > 82% on test set (slightly lower than BERT but more efficient)
   - F1 Score: > 0.82 on test set
   - Inference time: < 50ms per prompt (improved from 100ms)
   - Memory usage: < 1GB during inference (reduced from 2GB)
   - GPU utilization: > 80% during training

2. System Requirements
   - Docker container size: < 1.5GB (reduced from 2GB)
   - Memory usage: < 2GB (reduced from 4GB)
   - API response time: < 100ms (improved from 200ms)
   - CPU usage: < 30% during inference (reduced from 50%)
   - Disk space: < 3GB for model and data (reduced from 5GB)

## Risk Mitigation
1. Technical Risks
   - Data quality issues
     * Solution: Implement data validation pipeline
     * Tools: Great Expectations
   
   - Model overfitting
     * Solution: Regularization and early stopping
     * Tools: TensorBoard for monitoring
   
   - Performance bottlenecks
     * Solution: Profiling and optimization
     * Tools: PyTorch Profiler

2. Mitigation Strategies
   - Regular data validation
   - Cross-validation
   - Performance profiling
   - Regular backups
   - A/B testing for model updates
   - Canary deployments 