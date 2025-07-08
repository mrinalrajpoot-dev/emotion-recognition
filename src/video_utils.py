import cv2
import os
import numpy as np
from datetime import datetime

def draw_bounding_box_with_label(frame, box, label, confidence=None, color=(0, 255, 0), 
                                 thickness=2, font_scale=0.6):
    """
    Draw bounding box with label on frame.
    
    Args:
        frame: Input frame
        box: Bounding box coordinates (x, y, w, h)
        label: Text label to display
        confidence: Optional confidence score to display
        color: BGR color tuple
        thickness: Line thickness
        font_scale: Font size scale
        
    Returns:
        Frame with bounding box and label drawn
    """
    x, y, w, h = box
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Prepare label text
    if confidence is not None:
        text = f"{label}: {confidence:.2%}"
    else:
        text = label
    
    # Calculate text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw text background
    cv2.rectangle(frame, 
                  (x, y - text_height - 10), 
                  (x + text_width + 10, y), 
                  color, -1)
    
    # Draw text
    cv2.putText(frame, text, 
                (x + 5, y - 5), 
                font, font_scale, 
                (255, 255, 255), thickness - 1)
    
    return frame

def create_video_writer(output_path, fps=30, frame_size=None, codec='mp4v'):
    """
    Create video writer for saving output.
    
    Args:
        output_path: Path for output video file
        fps: Frames per second
        frame_size: Tuple of (width, height). If None, will be set on first write
        codec: Video codec (default: mp4v)
        
    Returns:
        cv2.VideoWriter object
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define codec
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # Create video writer
    if frame_size:
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    else:
        writer = None  # Will be initialized on first frame
    
    return writer, fourcc, fps

def write_frame(writer, frame, fourcc=None, fps=30, output_path=None):
    """
    Write frame to video file, initializing writer if needed.
    
    Args:
        writer: cv2.VideoWriter object or None
        frame: Frame to write
        fourcc: Video codec (required if writer is None)
        fps: Frames per second (required if writer is None)
        output_path: Output path (required if writer is None)
        
    Returns:
        Updated writer object
    """
    if writer is None:
        if fourcc is None or output_path is None:
            raise ValueError("fourcc and output_path required for first write")
        
        frame_size = (frame.shape[1], frame.shape[0])
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    writer.write(frame)
    return writer

def add_timestamp(frame, position='top-right', color=(255, 255, 255), 
                  font_scale=0.5, thickness=1):
    """
    Add timestamp to frame.
    
    Args:
        frame: Input frame
        position: Position of timestamp ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        color: Text color (BGR)
        font_scale: Font size scale
        thickness: Text thickness
        
    Returns:
        Frame with timestamp
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(timestamp, font, font_scale, thickness)
    
    # Calculate position
    margin = 10
    h, w = frame.shape[:2]
    
    positions = {
        'top-left': (margin, text_height + margin),
        'top-right': (w - text_width - margin, text_height + margin),
        'bottom-left': (margin, h - margin),
        'bottom-right': (w - text_width - margin, h - margin)
    }
    
    pos = positions.get(position, positions['top-right'])
    
    # Add background for better visibility
    cv2.rectangle(frame, 
                  (pos[0] - 5, pos[1] - text_height - 5),
                  (pos[0] + text_width + 5, pos[1] + 5),
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, timestamp, pos, font, font_scale, color, thickness)
    
    return frame

def add_fps_counter(frame, fps, position='top-left', color=(0, 255, 0)):
    """
    Add FPS counter to frame.
    
    Args:
        frame: Input frame
        fps: Current FPS value
        position: Position of counter
        color: Text color (BGR)
        
    Returns:
        Frame with FPS counter
    """
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate position
    margin = 10
    h, w = frame.shape[:2]
    
    if position == 'top-left':
        pos = (margin, text_height + margin)
    else:
        pos = (margin, text_height + margin)
    
    # Add background
    cv2.rectangle(frame,
                  (pos[0] - 5, pos[1] - text_height - 5),
                  (pos[0] + text_width + 5, pos[1] + 5),
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, pos, font, font_scale, color, thickness)
    
    return frame

def resize_frame(frame, width=None, height=None, maintain_aspect=True):
    """
    Resize frame to specified dimensions.
    
    Args:
        frame: Input frame
        width: Target width
        height: Target height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if width is None and height is None:
        return frame
    
    if maintain_aspect:
        if width is not None:
            aspect = w / h
            height = int(width / aspect)
        else:
            aspect = w / h
            width = int(height * aspect)
    else:
        if width is None:
            width = w
        if height is None:
            height = h
    
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def create_emotion_overlay(frame, emotions_dict, position='bottom', alpha=0.7):
    """
    Create an overlay showing emotion distribution.
    
    Args:
        frame: Input frame
        emotions_dict: Dictionary of emotions and their probabilities
        position: Position of overlay ('top', 'bottom', 'left', 'right')
        alpha: Transparency of overlay
        
    Returns:
        Frame with emotion overlay
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Sort emotions by probability
    sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Define overlay dimensions
    bar_height = 30
    overlay_height = len(sorted_emotions) * bar_height + 20
    overlay_width = 300
    
    # Define position
    if position == 'bottom':
        y_start = h - overlay_height - 10
    else:
        y_start = 10
    
    x_start = 10
    
    # Draw semi-transparent background
    cv2.rectangle(overlay, 
                  (x_start, y_start), 
                  (x_start + overlay_width, y_start + overlay_height),
                  (0, 0, 0), -1)
    
    # Draw emotion bars
    for i, (emotion, prob) in enumerate(sorted_emotions):
        y = y_start + 10 + i * bar_height
        
        # Draw bar
        bar_width = int(prob * (overlay_width - 100))
        cv2.rectangle(overlay,
                      (x_start + 80, y),
                      (x_start + 80 + bar_width, y + 20),
                      (0, 255, 0), -1)
        
        # Draw label
        cv2.putText(overlay, f"{emotion}:", 
                    (x_start + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw percentage
        cv2.putText(overlay, f"{prob:.1%}",
                    (x_start + overlay_width - 50, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend overlay with original frame
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def save_frame_snapshot(frame, output_dir='snapshots', prefix='emotion'):
    """
    Save a snapshot of the current frame.
    
    Args:
        frame: Frame to save
        output_dir: Directory to save snapshots
        prefix: Filename prefix
        
    Returns:
        Path to saved file
    """
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Save frame
    cv2.imwrite(filepath, frame)
    
    return filepath

class FPSCounter:
    """Simple FPS counter for video processing."""
    
    def __init__(self, update_interval=10):
        """
        Initialize FPS counter.
        
        Args:
            update_interval: Number of frames between FPS updates
        """
        self.update_interval = update_interval
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = cv2.getTickCount()
    
    def update(self):
        """Update FPS counter and return current FPS."""
        self.frame_count += 1
        
        if self.frame_count % self.update_interval == 0:
            end_time = cv2.getTickCount()
            time_diff = (end_time - self.start_time) / cv2.getTickFrequency()
            self.fps = self.update_interval / time_diff
            self.start_time = end_time
        
        return self.fps
    
    def reset(self):
        """Reset FPS counter."""
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = cv2.getTickCount()
