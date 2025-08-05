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

        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(1024, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.GlobalAveragePooling2D())

        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
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
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        shear_range=0.3,
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1,
        fill_mode='nearest'
    )

    datagen.fit(X_train)

    input_shape = (128, 128, 3)
    num_classes = len(class_names)

    model = create_enhanced_cnn(input_shape, num_classes)

    optimizer = optimizers.Adam(
        learning_rate=0.0005,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-8,
        verbose=1,
        cooldown=5
    )

    checkpoint = callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    )

    cosine_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.8,
        patience=5,
        min_lr=1e-9,
        verbose=1
    )

    print("\nTraining the model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        steps_per_epoch=len(X_train) // 16,
        epochs=150,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint, cosine_scheduler],
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

    model.save('recycleai_model.h5')
    model.save_weights('recycleai_model.weights.h5')

    with open('class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    print(f"\nModel saved to 'recycleai_model.h5'")
    print(f"Weights saved to 'recycleai_model.weights.h5'")
    print(f"Class names saved to 'class_names.txt'")

    plot(history)
