import cv2
import os
import numpy as np

class FaceDetector:
    def __init__(self, cascade_path=None, scale_factor=1.1, min_neighbors=5):
        """
        Initialize face detector with Haar Cascade classifier.
        
        Args:
            cascade_path: Path to cascade file. If None, uses OpenCV's default
            scale_factor: Parameter specifying how much the image size is reduced at each scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        
        # Load Haar Cascade
        if cascade_path is None:
            # Use OpenCV's built-in cascade file
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Failed to load cascade classifier")
    
    def detect_faces(self, frame, return_gray=False):
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input image/frame (BGR format)
            return_gray: If True, also returns the grayscale image
            
        Returns:
            List of tuples (x, y, w, h) representing face bounding boxes
            If return_gray=True, returns (faces, gray_frame)
        """
        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Histogram equalization for better detection in varying lighting
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert numpy array to list of tuples
        face_list = [(x, y, w, h) for (x, y, w, h) in faces]
        
        if return_gray:
            return face_list, gray
        return face_list
    
    def detect_faces_multiscale(self, frame, scales=[1.0, 0.75, 0.5]):
        """
        Detect faces at multiple scales for improved accuracy.
        
        Args:
            frame: Input image/frame
            scales: List of scales to try
            
        Returns:
            List of unique face bounding boxes
        """
        all_faces = []
        original_height, original_width = frame.shape[:2]
        
        for scale in scales:
            # Resize frame
            width = int(original_width * scale)
            height = int(original_height * scale)
            resized = cv2.resize(frame, (width, height))
            
            # Detect faces
            faces = self.detect_faces(resized)
            
            # Scale back coordinates
            for (x, y, w, h) in faces:
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
                all_faces.append((x, y, w, h))
        
        # Remove overlapping detections
        return self._non_max_suppression(all_faces)
    
    def _non_max_suppression(self, boxes, overlap_thresh=0.3):
        """
        Apply non-maximum suppression to remove overlapping bounding boxes.
        
        Args:
            boxes: List of bounding boxes (x, y, w, h)
            overlap_thresh: Overlap threshold for suppression
            
        Returns:
            List of filtered bounding boxes
        """
        if len(boxes) == 0:
            return []
        
        # Convert to numpy array
        boxes_array = np.array(boxes)
        
        # Get coordinates
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 0] + boxes_array[:, 2]
        y2 = boxes_array[:, 1] + boxes_array[:, 3]
        
        # Calculate areas
        areas = boxes_array[:, 2] * boxes_array[:, 3]
        
        # Sort by bottom-right y-coordinate
        indices = np.argsort(y2)
        
        keep = []
        while len(indices) > 0:
            # Take last index
            last = len(indices) - 1
            i = indices[last]
            keep.append(i)
            
            # Calculate overlap with all other boxes
            xx1 = np.maximum(x1[i], x1[indices[:last]])
            yy1 = np.maximum(y1[i], y1[indices[:last]])
            xx2 = np.minimum(x2[i], x2[indices[:last]])
            yy2 = np.minimum(y2[i], y2[indices[:last]])
            
            # Calculate intersection area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / areas[indices[:last]]
            
            # Remove indices with high overlap
            indices = np.delete(indices, 
                              np.concatenate(([last], 
                                            np.where(overlap > overlap_thresh)[0])))
        
        return [boxes[i] for i in keep]
    
    def draw_faces(self, frame, faces, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes around detected faces.
        
        Args:
            frame: Input frame to draw on
            faces: List of face bounding boxes
            color: BGR color for rectangles
            thickness: Rectangle thickness
            
        Returns:
            Frame with drawn bounding boxes
        """
        frame_copy = frame.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
        
        return frame_copy
    
    def get_face_roi(self, frame, face_coords, padding=0):
        """
        Extract face region of interest with optional padding.
        
        Args:
            frame: Input frame
            face_coords: Face bounding box (x, y, w, h)
            padding: Padding around face in pixels
            
        Returns:
            Face ROI as numpy array
        """
        x, y, w, h = face_coords
        height, width = frame.shape[:2]
        
        # Apply padding with bounds checking
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(width, x + w + padding)
        y_end = min(height, y + h + padding)
        
        return frame[y_start:y_end, x_start:x_end]
