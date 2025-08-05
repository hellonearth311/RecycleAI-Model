import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from preprocess import load_and_preprocess_data
from plot import plot

def create_cnn(input_shape, num_classes):
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# preprocess
print("Loading and preprocessing images...")
data_dir = "data"
X, y, class_names = load_and_preprocess_data(data_dir)

print(f"\nDataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

# compile the model
input_shape = (128, 128, 3)
num_classes = len(class_names)

model = create_cnn(input_shape, num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# evaluation
print("\nEvaluating model on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

correct_predictions = np.sum(y_pred_classes == y_test)
incorrect_predictions = len(y_test) - correct_predictions

# output
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

# plot
plot(history)
