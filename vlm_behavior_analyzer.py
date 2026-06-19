"""
VLM Behavior Analysis Module
Captures 5-second video clips at 3 FPS when suspicious objects are detected
and analyzes behavior using Vision Language Model (VLM)
"""

import cv2
import time
import os
from datetime import datetime
from pathlib import Path
from collections import deque
import base64
import json
from typing import List, Dict, Tuple
import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ Warning: requests not installed. VLM API calls disabled.")


class VLMBehaviorAnalyzer:
    def __init__(self, config, vlm_config=None):
        """
        Initialize VLM Behavior Analyzer.
        
        Args:
            config: Main system configuration
            vlm_config: VLM-specific configuration
        """
        self.config = config
        self.vlm_config = vlm_config or config.get('vlm', {})
        
        # VLM settings
        self.vlm_enabled = self.vlm_config.get('enabled', True)
        self.vlm_api_url = self.vlm_config.get('api_url', 'http://localhost:8000/analyze')
        self.vlm_model = self.vlm_config.get('model', 'gemini-2.5-flash-lite')
        self.vlm_timeout = self.vlm_config.get('timeout', 30)
        
        # Video capture settings
        self.capture_duration = self.vlm_config.get('capture_duration', 5)  # 5 seconds
        self.capture_fps = self.vlm_config.get('capture_fps', 3)  # 3 FPS
        self.total_frames = self.capture_duration * self.capture_fps  # 15 frames
        
        # Frame buffer for continuous capture
        self.frame_buffer = deque(maxlen=self.total_frames)
        self.last_capture_time = {}
        self.capture_cooldown = self.vlm_config.get('capture_cooldown', 30)  # 30 seconds
        
        # Create directory for VLM analysis
        self.vlm_clips_dir = Path("vlm_analysis_clips")
        self.vlm_clips_dir.mkdir(exist_ok=True)
        
        # Behavior analysis prompts
        self.analysis_prompt = self.vlm_config.get('analysis_prompt', 
            """You are a security analyst reviewing surveillance footage. Analyze this sequence of frames 
            and describe:
            1. What suspicious activity or behavior is occurring?
            2. What weapon or object is visible and how is it being used?
            3. Is there an immediate threat or concerning behavior?
            4. What action should security personnel take?
            
            Provide a concise but detailed analysis."""
        )
        
        # Statistics
        self.vlm_analysis_count = 0
        self.vlm_success_count = 0
        self.vlm_fail_count = 0
        
        print(f"✅ VLM Behavior Analyzer initialized")
        print(f"   Capture: {self.capture_duration}s @ {self.capture_fps} FPS = {self.total_frames} frames")
        print(f"   VLM Model: {self.vlm_model}")
    
    def should_trigger_vlm(self, object_type: str) -> bool:
        """
        Check if VLM analysis should be triggered for this object detection.
        
        Args:
            object_type: Type of detected object
            
        Returns:
            bool: True if VLM should be triggered
        """
        if not self.vlm_enabled:
            return False
        
        # Check cooldown
        current_time = time.time()
        if object_type in self.last_capture_time:
            elapsed = current_time - self.last_capture_time[object_type]
            if elapsed < self.capture_cooldown:
                return False
        
        self.last_capture_time[object_type] = current_time
        return True
    
    def add_frame_to_buffer(self, frame: np.ndarray):
        """
        Add frame to circular buffer for continuous capture.
        
        Args:
            frame: Video frame to add
        """
        # Resize frame for efficiency
        resized_frame = cv2.resize(frame, (640, 480))
        self.frame_buffer.append(resized_frame.copy())
    
    def capture_video_clip(self) -> List[np.ndarray]:
        """
        Get frames from buffer for VLM analysis.
        
        Returns:
            List of frames for analysis
        """
        if len(self.frame_buffer) < self.total_frames:
            print(f"⚠️ Buffer not full yet ({len(self.frame_buffer)}/{self.total_frames} frames)")
            return list(self.frame_buffer)
        
        return list(self.frame_buffer)
    
    def save_video_clip(self, frames: List[np.ndarray], object_type: str) -> str:
        """
        Save captured frames as video clip.
        
        Args:
            frames: List of frames to save
            object_type: Type of detected object
            
        Returns:
            Path to saved video file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.vlm_clips_dir / f"clip_{object_type}_{timestamp}.mp4"
        
        if not frames:
            return None
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.capture_fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"📹 Video clip saved: {video_path}")
        return str(video_path)
    
    def frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        """
        Convert frames to base64 encoded strings for API transmission.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of base64 encoded frame strings
        """
        base64_frames = []
        for frame in frames:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Convert to base64
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(base64_frame)
        
        return base64_frames
    
    def analyze_with_vlm(self, frames: List[np.ndarray], object_type: str, 
                         confidence: float) -> Dict:
        """
        Send frames to VLM for behavior analysis.
        
        Args:
            frames: List of video frames
            object_type: Type of detected object
            confidence: Detection confidence
            
        Returns:
            Dictionary containing VLM analysis results
        """
        if not REQUESTS_AVAILABLE:
            print("❌ Requests library not available for VLM API calls")
            return None
        
        if not frames:
            print("❌ No frames to analyze")
            return None
        
        self.vlm_analysis_count += 1
        
        try:
            # Convert frames to base64
            print(f"🔄 Converting {len(frames)} frames to base64...")
            base64_frames = self.frames_to_base64(frames)
            
            # Prepare API request
            payload = {
                'model': self.vlm_model,
                'frames': base64_frames,
                'prompt': self.analysis_prompt,
                'object_type': object_type,
                'confidence': confidence,
                'frame_count': len(frames),
                'fps': self.capture_fps,
                'duration': self.capture_duration
            }
            
            print(f"🤖 Sending to VLM for analysis...")
            print(f"   Object: {object_type} (confidence: {confidence:.2%})")
            print(f"   Frames: {len(frames)} @ {self.capture_fps} FPS")
            
            # Make API request
            response = requests.post(
                self.vlm_api_url,
                json=payload,
                timeout=self.vlm_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.vlm_success_count += 1
                
                print(f"✅ VLM Analysis Complete!")
                print(f"   Analysis: {result.get('analysis', 'No analysis')[:100]}...")
                
                return result
            else:
                print(f"❌ VLM API error: {response.status_code}")
                print(f"   Response: {response.text}")
                self.vlm_fail_count += 1
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏱️ VLM analysis timed out after {self.vlm_timeout}s")
            self.vlm_fail_count += 1
            return None
        except Exception as e:
            print(f"❌ VLM analysis error: {str(e)}")
            self.vlm_fail_count += 1
            return None
    
    def process_detection(self, frame: np.ndarray, object_type: str, 
                         confidence: float, alert_system=None) -> Dict:
        """
        Main processing function when suspicious object is detected.
        
        Args:
            frame: Current video frame
            object_type: Type of detected object
            confidence: Detection confidence
            alert_system: AlertSystem instance for sending notifications
            
        Returns:
            Dictionary containing processing results
        """
        result = {
            'triggered': False,
            'captured': False,
            'analyzed': False,
            'video_path': None,
            'vlm_result': None
        }
        
        # Check if VLM should be triggered
        if not self.should_trigger_vlm(object_type):
            print(f"⏳ Cooldown active for {object_type} detection")
            return result
        
        result['triggered'] = True
        print(f"\n🎯 VLM Analysis Triggered for: {object_type}")
        
        # Capture video clip from buffer
        frames = self.capture_video_clip()
        
        if not frames:
            print("❌ No frames available in buffer")
            return result
        
        result['captured'] = True
        print(f"✅ Captured {len(frames)} frames from buffer")
        
        # Save video clip
        video_path = self.save_video_clip(frames, object_type)
        result['video_path'] = video_path
        
        # Analyze with VLM
        vlm_result = self.analyze_with_vlm(frames, object_type, confidence)
        
        if vlm_result:
            result['analyzed'] = True
            result['vlm_result'] = vlm_result
            
            # Send enhanced alert with VLM analysis
            if alert_system:
                self.send_vlm_enhanced_alert(
                    alert_system=alert_system,
                    object_type=object_type,
                    confidence=confidence,
                    video_path=video_path,
                    vlm_analysis=vlm_result.get('analysis', 'Analysis failed')
                )
        
        return result
    
    def send_vlm_enhanced_alert(self, alert_system, object_type: str, 
                                confidence: float, video_path: str, 
                                vlm_analysis: str):
        """
        Send alert with VLM behavior analysis included.
        
        Args:
            alert_system: AlertSystem instance
            object_type: Type of detected object
            confidence: Detection confidence
            video_path: Path to captured video clip
            vlm_analysis: VLM analysis text
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Enhanced message with VLM analysis
        enhanced_message = f"""
ENHANCED SECURITY ALERT WITH BEHAVIOR ANALYSIS

Object Detected: {object_type.upper()}
Detection Confidence: {confidence:.2%}
Timestamp: {timestamp}

AI BEHAVIOR ANALYSIS:
{vlm_analysis}

Video Clip: {video_path}

This alert includes AI-powered behavior analysis for enhanced threat assessment.
Please review the video clip and take appropriate action.
"""
        
        print(f"\n📧 Sending VLM-enhanced alert...")
        print(f"   Analysis preview: {vlm_analysis[:200]}...")
        
        # Send alert through the alert system
        try:
            alert_system.send_alert(
            object_type=f"{object_type}_VLM_ANALYZED",
            confidence=confidence,
            image_path=video_path,
            timestamp=timestamp,
            message=vlm_analysis
             )

        except Exception as e:
            print(f"⚠️ Error sending enhanced alert: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """Get VLM analyzer statistics."""
        return {
            'total_analyses': self.vlm_analysis_count,
            'successful': self.vlm_success_count,
            'failed': self.vlm_fail_count,
            'success_rate': (self.vlm_success_count / max(1, self.vlm_analysis_count)) * 100
        }