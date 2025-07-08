import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Emotion labels mapping (FER2013 standard)
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Reverse mapping for convenience
LABEL_TO_INDEX = {v: k for k, v in EMOTION_LABELS.items()}

def preprocess_face(face_roi, target_size=(48, 48)):
    """
    Preprocess face ROI for emotion model input.
    
    Args:
        face_roi: Face region of interest (numpy array)
        target_size: Target size for the model input (width, height)
        
    Returns:
        Preprocessed face ready for model prediction (1, 48, 48, 1)
    """
    # Convert to grayscale if needed
    if len(face_roi.shape) == 3:
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_roi
    
    # Resize to target size
    face_resized = cv2.resize(face_gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Apply histogram equalization for better contrast
    face_resized = cv2.equalizeHist(face_resized)
    
    # Convert to array and normalize
    face_array = img_to_array(face_resized)
    face_array = face_array.astype('float32') / 255.0
    
    # Expand dimensions for model input (1, 48, 48, 1)
    face_array = np.expand_dims(face_array, axis=0)
    
    return face_array

def decode_emotion(prediction_idx):
    """
    Decode numeric prediction to emotion label.
    
    Args:
        prediction_idx: Numeric index from model prediction
        
    Returns:
        Emotion label string
    """
    return EMOTION_LABELS.get(prediction_idx, 'Unknown')

def decode_predictions(predictions, top_k=3):
    """
    Decode model predictions to emotion labels with confidence scores.
    
    Args:
        predictions: Model output probabilities (numpy array)
        top_k: Number of top predictions to return
        
    Returns:
        List of tuples (emotion_label, confidence_score) sorted by confidence
    """
    # Get indices sorted by confidence (descending)
    top_indices = np.argsort(predictions)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        emotion = decode_emotion(idx)
        confidence = predictions[idx]
        results.append((emotion, confidence))
    
    return results

def get_emotion_index(emotion_label):
    """
    Get numeric index for emotion label.
    
    Args:
        emotion_label: String emotion label
        
    Returns:
        Numeric index or -1 if not found
    """
    return LABEL_TO_INDEX.get(emotion_label, -1)

def augment_face(face_roi, augmentation_type='none'):
    """
    Apply data augmentation to face ROI for training.
    
    Args:
        face_roi: Face region of interest
        augmentation_type: Type of augmentation ('flip', 'rotate', 'brightness', 'none')
        
    Returns:
        Augmented face ROI
    """
    face = face_roi.copy()
    
    if augmentation_type == 'flip':
        face = cv2.flip(face, 1)  # Horizontal flip
    
    elif augmentation_type == 'rotate':
        angle = np.random.uniform(-10, 10)  # Random rotation between -10 and 10 degrees
        h, w = face.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        face = cv2.warpAffine(face, matrix, (w, h))
    
    elif augmentation_type == 'brightness':
        factor = np.random.uniform(0.7, 1.3)  # Random brightness adjustment
        face = cv2.convertScaleAbs(face, alpha=factor, beta=0)
    
    return face

def batch_preprocess_faces(face_list, target_size=(48, 48)):
    """
    Preprocess multiple face ROIs for batch prediction.
    
    Args:
        face_list: List of face ROIs
        target_size: Target size for model input
        
    Returns:
        Numpy array of preprocessed faces (batch_size, 48, 48, 1)
    """
    processed_faces = []
    
    for face_roi in face_list:
        processed = preprocess_face(face_roi, target_size)
        processed_faces.append(processed[0])  # Remove batch dimension
    
    return np.array(processed_faces)

def get_emotion_color(emotion):
    """
    Get color (BGR) associated with emotion for visualization.
    
    Args:
        emotion: Emotion label string
        
    Returns:
        BGR color tuple
    """
    emotion_colors = {
        'Angry': (0, 0, 255),      # Red
        'Disgust': (0, 100, 0),    # Dark Green
        'Fear': (255, 0, 255),     # Magenta
        'Happy': (0, 255, 255),    # Yellow
        'Neutral': (128, 128, 128), # Gray
        'Sad': (255, 0, 0),        # Blue
        'Surprise': (0, 255, 0)    # Green
    }
    
    return emotion_colors.get(emotion, (255, 255, 255))  # Default to white

def calculate_emotion_metrics(predictions, ground_truth):
    """
    Calculate accuracy metrics for emotion predictions.
    
    Args:
        predictions: List of predicted emotion indices
        ground_truth: List of ground truth emotion indices
        
    Returns:
        Dictionary with accuracy metrics
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Overall accuracy
    accuracy = np.mean(predictions == ground_truth)
    
    # Per-emotion accuracy
    per_emotion_accuracy = {}
    for emotion_idx, emotion_label in EMOTION_LABELS.items():
        mask = ground_truth == emotion_idx
        if np.sum(mask) > 0:
            per_emotion_accuracy[emotion_label] = np.mean(predictions[mask] == emotion_idx)
        else:
            per_emotion_accuracy[emotion_label] = 0.0
    
    return {
        'overall_accuracy': accuracy,
        'per_emotion_accuracy': per_emotion_accuracy
    }

def save_preprocessed_dataset(faces, labels, output_path):
    """
    Save preprocessed faces and labels for faster training.
    
    Args:
        faces: Numpy array of preprocessed faces
        labels: Numpy array of emotion labels
        output_path: Path to save the dataset
    """
    np.savez_compressed(output_path, faces=faces, labels=labels)
    print(f"Dataset saved to {output_path}")

def load_preprocessed_dataset(input_path):
    """
    Load preprocessed dataset.
    
    Args:
        input_path: Path to the saved dataset
        
    Returns:
        Tuple of (faces, labels)
    """
    data = np.load(input_path)
    return data['faces'], data['labels']
