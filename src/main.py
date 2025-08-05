import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from preprocess import load_and_preprocess_data
from plot import plot

tf.config.experimental.enable_op_determinism()

print("Configuring Metal Performance Shaders for Apple Silicon...")
try:
    physical_devices = tf.config.list_physical_devices()
    print(f"Available devices: {physical_devices}")
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Metal GPU devices found: {gpu_devices}")
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Metal Performance Shaders (MPS) enabled for M4 Pro GPU")
        device_name = '/GPU:0'
    else:
        print("No Metal GPU detected, using CPU")
        device_name = '/CPU:0'
        
except Exception as e:
    print(f"Metal setup error: {e}")
    print("Falling back to CPU")
    device_name = '/CPU:0'

with tf.device(device_name):
    def create_enhanced_cnn(input_shape, num_classes):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(num_classes, activation='softmax'))

        return model

    print("Loading and preprocessing images...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    print(f"Data directory path: {data_dir}")
    X, y, class_names = load_and_preprocess_data(data_dir)

    print(f"\nDataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    from sklearn.utils.class_weight import compute_class_weight
    from collections import Counter
    
    label_counts = Counter(y)
    print(f"Raw class distribution: {dict(label_counts)}")
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    
    min_samples = min(label_counts.values())
    max_samples = max(label_counts.values())
    ratio = max_samples / min_samples
    
    if ratio > 3:
        for i, weight in enumerate(class_weights):
            if label_counts[i] < (max_samples * 0.4):
                class_weights[i] *= 1.5
    
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    print(f"Adjusted class weights: {dict(zip(class_names, [class_weight_dict[i] for i in range(len(class_names))]))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTrain set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )

    datagen.fit(X_train)

    input_shape = (128, 128, 3)
    num_classes = len(class_names)

    model = create_enhanced_cnn(input_shape, num_classes)

    optimizer = optimizers.Adam(
        learning_rate=0.0005,
        beta_1=0.9,
        beta_2=0.999
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.01
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=6,
        min_lr=1e-7,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    )

    print("\nTraining the model...")
    
    def balanced_batch_generator(X, y, batch_size, datagen, class_weights):
        while True:
            # Create proper probability distribution
            weights_per_sample = np.array([class_weights[label] for label in y])
            probabilities = weights_per_sample / np.sum(weights_per_sample)
            
            indices = np.random.choice(
                len(X), 
                size=batch_size, 
                replace=True,
                p=probabilities
            )
            
            batch_X = X[indices].copy()
            batch_y = y[indices]
            
            # Apply data augmentation
            for i in range(len(batch_X)):
                batch_X[i] = datagen.random_transform(batch_X[i])
            
            yield batch_X, batch_y
    
    history = model.fit(
        balanced_batch_generator(X_train, y_train, 32, datagen, class_weight_dict),
        steps_per_epoch=len(X_train) // 32,
        epochs=75,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    correct_predictions = np.sum(y_pred_classes == y_test)
    incorrect_predictions = len(y_test) - correct_predictions

    print(f"\n{'='*50}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Total Test Images: {len(y_test)}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Incorrect Predictions: {incorrect_predictions}")
    print(f"{'='*50}")

    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum(y_pred_classes[class_mask] == y_test[class_mask]) / np.sum(class_mask)
            print(f"{class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred_classes)
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))

    model.save('recycleai_model.h5')
    model.save_weights('recycleai_model.weights.h5')

    with open('class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    print(f"\nModel saved to 'recycleai_model.h5'")
    print(f"Weights saved to 'recycleai_model.weights.h5'")
    print(f"Class names saved to 'class_names.txt'")

    plot(history)
