import os
import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

def load_and_preprocess_data(data_dir, target_size=(128, 128)):
    images = []
    labels = []
    
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found classes: {class_names}")
    
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        class_idx = class_to_idx[class_name]
        
        print(f"Loading images from {class_name}...")
        
        class_images = []
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = Image.open(img_path)
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    class_images.append(img_array)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        print(f"Loaded {len(class_images)} images from {class_name}")
        
        images.extend(class_images)
        labels.extend([class_idx] * len(class_images))
    
    print(f"Total images loaded: {len(images)}")
    class_counts = [labels.count(i) for i in range(len(class_names))]
    print(f"Images per class: {dict(zip(class_names, class_counts))}")
    
    return np.array(images, dtype=np.float32), np.array(labels), class_names