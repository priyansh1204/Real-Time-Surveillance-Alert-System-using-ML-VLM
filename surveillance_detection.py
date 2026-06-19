
"""
Real-Time Surveillance Alert System with VLM Behavior Analysis
Detects suspicious objects using YOLOv8 and analyzes behavior using VLM
"""

import cv2
import time
import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import json
from alert_system import AlertSystem
from vlm_behavior_analyzer import VLMBehaviorAnalyzer


class SurveillanceDetector:
    def __init__(self, model_path, config_path="config.json"):
        """
        Initialize the surveillance detector with VLM integration.
        
        Args:
            model_path: Path to YOLOv8 .pt model file
            config_path: Path to configuration JSON file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize YOLOv8 model
        print(f"Loading YOLOv8 model from {model_path}...")
        try:
            self.model = YOLO(model_path)
            print("✅ Model loaded successfully!")
            print(f"   Model has {len(self.model.names)} classes")
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            raise
        
        # Initialize alert system
        try:
            self.alert_system = AlertSystem(self.config)
            print("✅ Alert system initialized")
        except Exception as e:
            print(f"⚠️ Alert system initialization warning: {str(e)}")
            print("   Continuing without alert system...")
            self.alert_system = None
        
        # Initialize VLM Behavior Analyzer
        try:
            self.vlm_analyzer = VLMBehaviorAnalyzer(self.config)
            print("✅ VLM Behavior Analyzer initialized")
        except Exception as e:
            print(f"⚠️ VLM analyzer initialization warning: {str(e)}")
            print("   Continuing without VLM analysis...")
            self.vlm_analyzer = None
        
        # Suspicious object classes
        self.suspicious_classes = self.config.get('suspicious_classes', [
            'gun', 'knife', 'rod', 'stick', 'wooden_stick', 'weapon', 
            'pistol', 'rifle', 'blade', 'blunt_object'
        ])
        print(f"✅ Monitoring {len(self.suspicious_classes)} suspicious object types")
        
        # Detection settings
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.alert_cooldown = self.config.get('alert_cooldown', 10)
        self.last_alert_time = {}
        
        # Create alerts directory
        self.alerts_dir = Path("alerts")
        self.alerts_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.detection_count = 0
        self.alert_count = 0
        self.vlm_trigger_count = 0
        
        # Check GUI availability
        self.gui_available = self.check_gui_available()
        if self.gui_available:
            print("✅ GUI display available")
        else:
            print("⚠️ GUI display not available - will run in headless mode")
        
    def load_config(self, config_path):
        """Load configuration from JSON file."""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"✅ Configuration loaded from {config_path}")
                return config
            except Exception as e:
                print(f"⚠️ Error loading config: {str(e)}")
                return {}
        else:
            print(f"⚠️ Config file not found: {config_path}")
        return {}
    
    def check_gui_available(self):
        """Check if OpenCV GUI functions are available."""
        try:
            import numpy as np
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.imshow('test', test_img)
            cv2.waitKey(1)
            cv2.destroyWindow('test')
            cv2.destroyAllWindows()
            return True
        except (cv2.error, Exception):
            return False
    
    def is_suspicious_object(self, class_name):
        """Check if detected object is in suspicious classes list."""
        class_lower = class_name.lower()
        for suspicious in self.suspicious_classes:
            if suspicious.lower() in class_lower or class_lower in suspicious.lower():
                return True
        return False
    
    def should_send_alert(self, class_name):
        """Check if alert should be sent (cooldown mechanism)."""
        current_time = time.time()
        if class_name not in self.last_alert_time:
            self.last_alert_time[class_name] = 0
        
        if current_time - self.last_alert_time[class_name] >= self.alert_cooldown:
            self.last_alert_time[class_name] = current_time
            return True
        return False
    
    def save_alert_image(self, frame, detections):
        """Save frame with detections as alert evidence."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_image_path = self.alerts_dir / f"alert_{timestamp}.jpg"
        
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} ({det['confidence']:.2f})"
            
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imwrite(str(alert_image_path), annotated_frame)
        return alert_image_path
    
    def process_frame(self, frame):
        """
        Process a single frame and detect suspicious objects.
        
        Returns:
            tuple: (detections_list, annotated_frame)
        """
        results = self.model(
            frame, 
            conf=self.confidence_threshold, 
            verbose=False,
            imgsz=640,
            half=False
        )
        
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                if self.is_suspicious_object(class_name):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return detections, annotated_frame
    
    def run_webcam_detection(self, camera_index=0):
        """
        Run real-time detection on webcam feed with VLM integration.
        
        Args:
            camera_index: Index of camera device (default: 0)
        """
        print(f"\n🎥 Starting webcam detection with VLM analysis (Camera {camera_index})...")
        if self.gui_available:
            print("✅ GUI display enabled - Video window will be shown")
            print("Press 'q' or ESC to quit\n")
        else:
            print("⚠️ GUI display not available. Running in headless mode.")
            print("   Press Ctrl+C to stop\n")
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📹 Camera resolution: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        # Frame processing settings
        self.frame_skip = self.config.get('frame_skip', 1)
        self.frame_counter = 0
        
        # VLM buffer management
        vlm_buffer_frame_skip = max(1, int(actual_fps / 3))  # Capture at ~3 FPS for VLM
        vlm_frame_counter = 0
        
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Add frame to VLM buffer at 3 FPS
                vlm_frame_counter += 1
                if vlm_frame_counter % vlm_buffer_frame_skip == 0 and self.vlm_analyzer:
                    self.vlm_analyzer.add_frame_to_buffer(frame)
                
                # Frame skipping for object detection
                self.frame_counter += 1
                if self.frame_counter % self.frame_skip != 0:
                    continue
                
                # Process frame for object detection
                detections, annotated_frame = self.process_frame(frame)
                
                # Handle detections
                if detections:
                    self.detection_count += len(detections)
                    
                    for det in detections:
                        class_name = det['class']
                        
                        # Check if regular alert should be sent
                        if self.should_send_alert(class_name):
                            try:
                                alert_image_path = self.save_alert_image(frame, detections)
                                
                                # Trigger VLM analysis
                                if self.vlm_analyzer:
                                    print(f"\n{'='*60}")
                                    print(f"🎯 SUSPICIOUS OBJECT DETECTED: {class_name.upper()}")
                                    print(f"{'='*60}")
                                    
                                    vlm_result = self.vlm_analyzer.process_detection(
                                        frame=frame,
                                        object_type=class_name,
                                        confidence=det['confidence'],
                                        alert_system=self.alert_system
                                    )
                                    
                                    if vlm_result['triggered']:
                                        self.vlm_trigger_count += 1
                                        print(f"✅ VLM Analysis triggered ({self.vlm_trigger_count} total)")
                                    
                                    if vlm_result['analyzed']:
                                        print(f"✅ Behavior analysis complete!")
                                        if vlm_result['vlm_result']:
                                            analysis = vlm_result['vlm_result'].get('analysis', '')
                                            print(f"\n📊 VLM ANALYSIS:")
                                            print(f"{analysis[:300]}...")
                                    
                                    print(f"{'='*60}\n")
                                else:
                                    # Send regular alert if VLM not available
                                    if self.alert_system:
                                        self.alert_system.send_alert(
                                            object_type=class_name,
                                            confidence=det['confidence'],
                                            image_path=str(alert_image_path),
                                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        )
                                
                                self.alert_count += 1
                                print(f"🚨 ALERT: {class_name} detected! (Confidence: {det['confidence']:.2f})")
                                
                            except Exception as e:
                                print(f"⚠️ Error processing detection: {str(e)}")
                
                # Calculate FPS
                fps_frame_count += 1
                if fps_frame_count >= 30:
                    elapsed = time.time() - fps_start_time
                    if elapsed > 0:
                        fps = 30 / elapsed
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                # Display statistics
                stats_text = [
                    f"FPS: {fps:.1f}",
                    f"Detections: {self.detection_count}",
                    f"Alerts: {self.alert_count}",
                    f"VLM Triggers: {self.vlm_trigger_count}",
                    f"Status: {'ALERT' if detections else 'MONITORING'}"
                ]
                
                y_offset = 30
                for text in stats_text:
                    cv2.putText(annotated_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                # Display frame
                if self.gui_available:
                    try:
                        cv2.namedWindow('Surveillance Detection System (VLM Enabled)', cv2.WINDOW_NORMAL)
                        cv2.imshow('Surveillance Detection System (VLM Enabled)', annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            break
                    except cv2.error as e:
                        print(f"⚠️ GUI error: {e}")
                        self.gui_available = False
                else:
                    if fps_frame_count % 30 == 0:
                        print(f"FPS: {fps:.1f} | Det: {self.detection_count} | Alerts: {self.alert_count} | VLM: {self.vlm_trigger_count}")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            cap.release()
            if self.gui_available:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            
            print(f"\n✅ Detection stopped.")
            print(f"📊 FINAL STATISTICS:")
            print(f"   Total detections: {self.detection_count}")
            print(f"   Alerts sent: {self.alert_count}")
            print(f"   VLM analyses triggered: {self.vlm_trigger_count}")
            
            if self.vlm_analyzer:
                vlm_stats = self.vlm_analyzer.get_statistics()
                print(f"\n🤖 VLM ANALYSIS STATISTICS:")
                print(f"   Total analyses: {vlm_stats['total_analyses']}")
                print(f"   Successful: {vlm_stats['successful']}")
                print(f"   Failed: {vlm_stats['failed']}")
                print(f"   Success rate: {vlm_stats['success_rate']:.1f}%")


def main():
    """Main function to run the surveillance detection system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Surveillance with VLM Behavior Analysis')
    parser.add_argument('--model', type=str, default=r"D:\miner_project_all_material\vlm\best (epoch_100).pt",
                       help='Path to YOLOv8 model file (.pt)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--force-gui', action='store_true',
                       help='Force GUI mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    # Initialize detector
    detector = SurveillanceDetector(args.model, args.config)
    
    if args.force_gui:
        detector.gui_available = True
        print("🖥️ GUI mode forced on")
    
    # Run detection
    detector.run_webcam_detection(args.camera)


if __name__ == "__main__":
    main()