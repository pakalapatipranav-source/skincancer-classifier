import os
import json
import pickle
import argparse
import logging
import numpy as np
from datetime import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

tf.keras.backend.clear_session()

# === PARAMETERS ===
IMG_HEIGHT = 224
IMG_WIDTH = 224
DATA_DIR = "SkinCancerDS"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# === LOAD DATASET ===
def load_images(folder):
    images, labels = [], []
    for cls in os.listdir(folder):
        cls_folder = os.path.join(folder, cls)
        if not os.path.isdir(cls_folder):
            continue
        for f in os.listdir(cls_folder):
            path = os.path.join(cls_folder, f)
            try:
                img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='rgb')
                arr = img_to_array(img)
                # Use EfficientNet's preprocess_input instead of simple /255.0 normalization
                arr = preprocess_input(arr)
                images.append(arr)
                labels.append(cls)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
    return np.array(images), np.array(labels)

# === DATA AUGMENTATION ===
def create_data_augmentation():
    """Create data augmentation generator for training"""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def build_model(num_classes, freeze_base=True, learning_rate=0.001, base_model_ref=None):
    """
    Build the model architecture.
    
    Args:
        num_classes: Number of output classes
        freeze_base: Whether to freeze base model layers
        learning_rate: Learning rate for optimizer
        base_model_ref: Optional reference to existing base model (for Phase 2)
    
    Returns:
        Tuple of (compiled model, base_model reference)
    """
    # Base model - reuse if provided (for Phase 2), otherwise create new
    if base_model_ref is None:
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            pooling=None
        )
    else:
        base_model = base_model_ref
    
    # Freeze or unfreeze base model
    base_model.trainable = not freeze_base
    if freeze_base:
        logger.info("Base model layers are FROZEN")
    else:
        logger.info("Base model layers are TRAINABLE")
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - 30
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        logger.info(f"Fine-tuning from layer {fine_tune_at} onwards")
    
    # Build model with improved head
    inputs = base_model.input
    x = base_model.output
    
    # Use GlobalAveragePooling2D instead of Flatten (more efficient)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    # Compile with proper optimizer
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def main(epochs=10, batch_size=32, data_dir=DATA_DIR, use_augmentation=True, two_phase=True):
    """
    Main training function.
    
    Args:
        epochs: Number of training epochs per phase
        batch_size: Batch size for training
        data_dir: Directory containing train and test folders
        use_augmentation: Whether to use data augmentation
        two_phase: Whether to use two-phase training (frozen then unfrozen)
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    logger.info("Starting model training...")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, data_dir={data_dir}")
    
    # Load train/test sets
    logger.info("Loading training images...")
    X_train, y_train = load_images(train_dir)
    logger.info(f"Loaded {len(X_train)} training images")
    
    logger.info("Loading test images...")
    X_test, y_test = load_images(test_dir)
    logger.info(f"Loaded {len(X_test)} test images")

    # One-hot encode labels
    logger.info("Encoding labels...")
    lb = LabelBinarizer()
    y_train_encoded = lb.fit_transform(y_train)
    y_test_encoded = lb.transform(y_test)
    
    num_classes_lb = len(lb.classes_)
    logger.info(f"LabelBinarizer found {num_classes_lb} classes: {lb.classes_}")
    logger.info(f"Initial encoded shape: {y_train_encoded.shape}")
    
    # LabelBinarizer returns 1D array or shape (n, 1) for binary classification
    # Convert to proper one-hot encoding (n, 2) for binary classification
    if y_train_encoded.ndim == 1:
        # 1D array: [0, 1, 0, 1, ...]
        y_train_encoded = to_categorical(y_train_encoded, num_classes=num_classes_lb)
        y_test_encoded = to_categorical(y_test_encoded, num_classes=num_classes_lb)
    elif y_train_encoded.shape[1] == 1:
        # Shape (n, 1): needs to be flattened and converted
        y_train_encoded = to_categorical(y_train_encoded.flatten(), num_classes=num_classes_lb)
        y_test_encoded = to_categorical(y_test_encoded.flatten(), num_classes=num_classes_lb)
    
    logger.info(f"Label shape after encoding: {y_train_encoded.shape}")
    logger.info(f"Unique classes found: {lb.classes_}")
    
    # Train/validation split
    logger.info("Splitting training data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)
    
    # Verify label encoding
    num_classes = y_train.shape[1]
    logger.info(f"Number of classes (from label shape): {num_classes}")
    logger.info(f"y_train shape: {y_train.shape}, unique values: {np.unique(y_train)}")
    logger.info(f"y_val shape: {y_val.shape}, unique values: {np.unique(y_val)}")
    
    # Ensure we have the correct number of classes
    if num_classes != num_classes_lb:
        raise ValueError(f"Mismatch: LabelBinarizer found {num_classes_lb} classes, but encoded labels have shape indicating {num_classes} classes")
    
    # Class weights for imbalanced dataset
    y_labels = np.argmax(y_train, axis=1)
    unique_labels = np.unique(y_labels)
    logger.info(f"Unique label indices in training set: {unique_labels}")
    
    # Count samples per class
    class_counts = {i: np.sum(y_labels == i) for i in unique_labels}
    logger.info(f"Training samples per class: {class_counts}")
    
    # Compute balanced class weights
    cw = class_weight.compute_class_weight('balanced', classes=unique_labels, y=y_labels)
    cw_dict = dict(enumerate(cw))
    logger.info(f"Computed class weights: {cw_dict}")
    
    # AGGRESSIVE FIX: Since model keeps predicting only one class, use much stronger weights
    # Calculate inverse frequency weights manually for more control
    total_samples = len(y_labels)
    max_count = max(class_counts.values())
    
    # Use inverse frequency weighting with amplification
    logger.warning("⚠️  Applying AGGRESSIVE class weights to fix single-class prediction issue...")
    aggressive_cw_dict = {}
    for cls in unique_labels:
        # Inverse frequency: more samples = lower weight
        freq = class_counts[cls] / total_samples
        inverse_freq = 1.0 / freq
        # Amplify by 2x to strongly penalize majority class
        aggressive_cw_dict[cls] = inverse_freq * 2.0
        class_name = lb.classes_[cls] if hasattr(lb, 'classes_') and cls < len(lb.classes_) else f"class_{cls}"
        logger.info(f"  Class {cls} ({class_name}): {class_counts[cls]} samples ({freq*100:.1f}%) -> weight {aggressive_cw_dict[cls]:.3f}")
    
    cw_dict = aggressive_cw_dict
    logger.info(f"Final aggressive class weights: {cw_dict}")
    
    # Update y_test
    y_test = y_test_encoded
    
    # === SETUP CALLBACKS ===
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # === PHASE 1: TRAIN WITH FROZEN BASE ===
    logger.info("=" * 60)
    logger.info("PHASE 1: Training with frozen base model")
    logger.info("=" * 60)
    
    # Start with lower learning rate for better stability
    # Use even lower LR to prevent the model from getting stuck
    model, base_model = build_model(num_classes, freeze_base=True, learning_rate=0.0001)
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    # Log model output shape to verify it matches expected classes
    sample_input = np.random.random((1, IMG_HEIGHT, IMG_WIDTH, 3))
    sample_output = model.predict(sample_input, verbose=0)
    logger.info(f"Model output shape verification: {sample_output.shape} (expected: (1, {num_classes}))")
    
    # Prepare data
    if use_augmentation:
        logger.info("Using data augmentation for training")
        datagen = create_data_augmentation()
        train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    else:
        train_generator = None
    
    # Train Phase 1
    if use_augmentation:
        steps_per_epoch = len(X_train) // batch_size
        history1 = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            class_weight=cw_dict,
            verbose=1
        )
    else:
        history1 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=cw_dict,
            verbose=1
        )
    
    # === PHASE 2: FINE-TUNE WITH UNFROZEN BASE ===
    if two_phase:
        logger.info("=" * 60)
        logger.info("PHASE 2: Fine-tuning with unfrozen base model")
        logger.info("=" * 60)
        
        # Use the stored base_model reference from Phase 1
        # The base_model layers are shared with the model, so modifying base_model
        # will affect the layers in the built model
        logger.info("Unfreezing EfficientNet base model layers for fine-tuning...")
        
        # Unfreeze base model for fine-tuning
        base_model.trainable = True
        
        # Fine-tune from this layer onwards (unfreeze last 30 layers)
        fine_tune_at = len(base_model.layers) - 30
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        logger.info(f"Fine-tuning from layer {fine_tune_at} onwards (total layers: {len(base_model.layers)})")
        logger.info(f"Trainable layers in base: {sum([1 for layer in base_model.layers if layer.trainable])}")
        
        # Count total trainable parameters
        trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        logger.info(f"Total trainable params: {trainable_count:,}")
        logger.info(f"Total non-trainable params: {non_trainable_count:,}")
        
        # Recompile with even lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=0.00001),  # 10x lower for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model recompiled with lower learning rate (0.00001) for fine-tuning")
        
        # Continue training with lower learning rate
        # Reset callbacks for Phase 2 (fresh patience counters)
        callbacks_phase2 = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model_phase2.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        if use_augmentation:
            steps_per_epoch = len(X_train) // batch_size
            history2 = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks_phase2,
                class_weight=cw_dict,
                verbose=1
            )
        else:
            history2 = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_phase2,
                class_weight=cw_dict,
                verbose=1
            )
        
        # Combine histories
        history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
    else:
        history = history1.history
    
    # === EVALUATE ===
    logger.info("=" * 60)
    logger.info("Evaluating on test set...")
    logger.info("=" * 60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Detailed evaluation metrics
    y_pred = model.predict(X_test, verbose=0)
    logger.info(f"Model prediction shape: {y_pred.shape}")
    logger.info(f"Model output range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # Analyze prediction distribution
    logger.info(f"Sample predictions (first 10): {y_pred[:10]}")
    logger.info(f"Mean prediction per class: {y_pred.mean(axis=0)}")
    logger.info(f"Std prediction per class: {y_pred.std(axis=0)}")
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Check for class mismatch
    unique_pred = np.unique(y_pred_classes)
    unique_true = np.unique(y_test_classes)
    logger.info(f"Unique predicted classes: {unique_pred}")
    logger.info(f"Unique true classes: {unique_true}")
    
    # Count predictions per class
    pred_counts = {i: np.sum(y_pred_classes == i) for i in range(num_classes)}
    true_counts = {i: np.sum(y_test_classes == i) for i in range(num_classes)}
    logger.info(f"Prediction counts per class: {pred_counts}")
    logger.info(f"True counts per class: {true_counts}")
    
    # Check if model is always predicting same class
    if len(unique_pred) == 1:
        logger.warning(f"⚠️  MODEL ONLY PREDICTING CLASS {unique_pred[0]} - Model may not be learning!")
        logger.warning("This suggests:")
        logger.warning("1. Class imbalance too severe")
        logger.warning("2. Model not learning (check training metrics)")
        logger.warning("3. Learning rate may be too high/low")
        logger.warning("4. Data may have issues")
    
    # Classification report
    logger.info("\nClassification Report:")
    # Ensure all classes are represented in the labels parameter
    all_classes = np.arange(len(lb.classes_))
    logger.info("\n" + classification_report(
        y_test_classes, 
        y_pred_classes, 
        labels=all_classes,
        target_names=lb.classes_
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    
    # Per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    logger.info("\nPer-class Accuracies:")
    for i, (cls_name, acc) in enumerate(zip(lb.classes_, class_accuracies)):
        logger.info(f"  {cls_name}: {acc:.4f}")
    
    # === SAVE MODEL ===
    logger.info("Saving model...")
    model.save("skin_cancer_model.h5")
    logger.info("Model saved as skin_cancer_model.h5")
    
    # === SAVE LABEL BINARIZER ===
    with open("label_binarizer.pkl", "wb") as f:
        pickle.dump(lb, f)
    logger.info("Label binarizer saved as label_binarizer.pkl")
    
    # === SAVE CLASS NAMES ===
    class_names = lb.classes_.tolist()
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)
    logger.info(f"Class names saved as class_names.json: {class_names}")
    
    # === SAVE MODEL METADATA ===
    metadata = {
        "training_date": datetime.now().isoformat(),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "num_classes": len(class_names),
        "class_names": class_names,
        "input_shape": (IMG_HEIGHT, IMG_WIDTH, 3),
        "base_model": "EfficientNetB0",
        "epochs": epochs * (2 if two_phase else 1),
        "batch_size": batch_size,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "use_augmentation": use_augmentation,
        "two_phase_training": two_phase,
        "class_weights": {str(k): float(v) for k, v in cw_dict.items()},
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": {cls: float(acc) for cls, acc in zip(class_names, class_accuracies)}
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Model metadata saved as model_metadata.json")
    
    # Save training history
    if isinstance(history, dict):
        history_dict = {
            "loss": [float(x) for x in history['loss']],
            "accuracy": [float(x) for x in history['accuracy']],
            "val_loss": [float(x) for x in history['val_loss']],
            "val_accuracy": [float(x) for x in history['val_accuracy']]
        }
    else:
        history_dict = {
            "loss": [float(x) for x in history.history['loss']],
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_loss": [float(x) for x in history.history['val_loss']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']]
        }
    with open("training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Training history saved as training_history.json")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train skin cancer detection model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs per phase (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help=f'Directory containing train/test folders (default: {DATA_DIR})')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--no-two-phase', action='store_true', help='Disable two-phase training (only train frozen)')
    
    args = parser.parse_args()
    
    main(
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        data_dir=args.data_dir,
        use_augmentation=not args.no_augmentation,
        two_phase=not args.no_two_phase
    )
