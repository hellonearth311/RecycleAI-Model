import tensorflow as tf
import numpy as np
from PIL import Image
import os

def load_trained_model(model_path='recycleai_model.h5', class_names_path='class_names.txt'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first by running main.py")
    
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file not found: {class_names_path}. Please train the model first by running main.py")
    
    model = tf.keras.models.load_model(model_path)
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return model, class_names

def load_model_with_weights(weights_path='recycleai_model.weights.h5', class_names_path='class_names.txt'):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}. Please train the model first by running main.py")
    
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file not found: {class_names_path}. Please train the model first by running main.py")
    
    import sys
    sys.path.append('.')
    from main import create_cnn
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    model = create_cnn((128, 128, 3), len(class_names))
    model.load_weights(weights_path)
    
    return model, class_names

def predict_image(model, class_names, image_path):
    img = Image.open(image_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return class_names[predicted_class_idx], confidence

if __name__ == "__main__":
    try:
        model, class_names = load_trained_model()
        print(f"Model loaded successfully!")
        print(f"Classes: {class_names}")
        
        image_path = input("Enter path to image: ")
        try:
            predicted_class, confidence = predict_image(model, class_names, image_path)
            print(f"Prediction: {predicted_class}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        except Exception as e:
            print(f"Error predicting image: {e}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python main.py' first to train and save the model.")
    except Exception as e:
        print(f"Unexpected error: {e}")
