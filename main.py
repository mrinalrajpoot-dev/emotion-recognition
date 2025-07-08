import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector import FaceDetector
from emotion_utils import decode_emotion, preprocess_face

class EmotionRecognizer:
    def __init__(self, model_path='emotion_model.h5'):
        """Initialize the emotion recognizer with model and face detector."""
        self.model = load_model(model_path)
        self.face_detector = FaceDetector()
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
    def predict_emotion(self, face_roi):
        """Predict emotion from face ROI."""
        # Preprocess face for model input
        processed_face = preprocess_face(face_roi, target_size=(48, 48))
        
        # Make prediction
        prediction = self.model.predict(processed_face, verbose=0)
        emotion_idx = np.argmax(prediction[0])
        confidence = prediction[0][emotion_idx]
        
        return self.emotions[emotion_idx], confidence
    
    def draw_results(self, frame, face_coords, emotion, confidence):
        """Draw bounding box and emotion label on frame."""
        x, y, w, h = face_coords
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare label with emotion and confidence
        label = f"{emotion}: {confidence:.2f}"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
        return frame

def main():
    """Main function to run emotion recognition on webcam feed."""
    # Initialize emotion recognizer
    try:
        recognizer = EmotionRecognizer()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Detect faces
        faces = recognizer.face_detector.detect_faces(frame)
        
        # Process each detected face
        for face_coords in faces:
            x, y, w, h = face_coords
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Predict emotion
                emotion, confidence = recognizer.predict_emotion(face_roi)
                
                # Draw results on frame
                frame = recognizer.draw_results(frame, face_coords, emotion, confidence)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Display FPS
        cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Emotion Recognition', frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully")

if __name__ == "__main__":
    main()
