"""
VLM API Server - Gemini 2.5 Flash Integration
Flask server to handle VLM behavior analysis using Google Gemini
"""

from flask import Flask, request, jsonify
import base64
import io
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import os
import tempfile
from pathlib import Path

# Google Gemini imports
from google import genai
from google.genai.types import Part

app = Flask(__name__)

# Initialize Gemini client
try:
    client = genai.Client(vertexai=True, project="pristine-sphere-477402-t2")
    print("✅ Gemini client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Gemini client: {str(e)}")
    client = None


def base64_to_frame(base64_str):
    """Convert base64 string to OpenCV frame."""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame


def frame_to_bytes(frame):
    """Convert OpenCV frame to bytes."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()


def analyze_frame_with_gemini(frame_bytes, prompt):
    """
    Analyze a single frame with Gemini 2.5 Flash.
    
    Args:
        frame_bytes: Image bytes
        prompt: Analysis prompt
        
    Returns:
        str: Response from Gemini
    """
    try:
        # Create Part object from bytes
        image_part = Part.from_bytes(
            data=frame_bytes,
            mime_type='image/jpeg'
        )
        
        # Send to Gemini
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[image_part, prompt],
        )
        
        return response.text.strip()
        
    except Exception as e:
        print(f"❌ Gemini analysis error: {str(e)}")
        return None


def analyze_frames_sequence(frames, object_type, confidence):
    """
    Analyze sequence of frames for suspicious behavior.
    Uses two-stage analysis:
    1. Quick check on selected frames (yes/no)
    2. Detailed analysis if suspicious activity detected
    
    Args:
        frames: List of OpenCV frames
        object_type: Detected object type
        confidence: Detection confidence
        
    Returns:
        Dictionary with analysis results
    """
    if not client:
        return {
            'success': False,
            'error': 'Gemini client not initialized',
            'analysis': 'VLM service unavailable'
        }
    
    print(f"🔍 Analyzing {len(frames)} frames with Gemini 2.5 Flash...")
    
    # Stage 1: Quick suspicious activity check on sampled frames
    # Sample 3 key frames (beginning, middle, end)
    sample_indices = [0, len(frames)//2, len(frames)-1] if len(frames) >= 3 else [0]
    suspicious_detected = False
    
    quick_check_prompt = """You will be provided with an image for analysis:

Follow these instructions precisely:
1. Carefully analyze the content of the provided image.
2. Determine if the image contains any form of suspicious activity. Suspicious activity includes, but is not limited to:
   * Violent scenes (e.g., fighting, physical assault, visible injuries from violence).
   * Depiction of weapons being used in a threatening manner.
   * Any clear depiction of a criminal act (e.g., theft, vandalism, arson).
   * Threatening gestures or aggressive behavior.
   * Any activity that poses immediate danger to people or property.
3. Based on your analysis, provide your response according to the following rules:
   * If the image contains any suspicious activity, your response MUST be the single word: `yes`.
   * If the image does not contain any suspicious activity, your response MUST be the single word: `no`.

Your entire response must be ONLY 'yes' or 'no'. Do not include any other words, explanations, or punctuation."""

    print(f"🔎 Stage 1: Quick check on {len(sample_indices)} key frames...")
    
    for idx in sample_indices:
        frame = frames[idx]
        frame_bytes = frame_to_bytes(frame)
        
        result = analyze_frame_with_gemini(frame_bytes, quick_check_prompt)
        
        if result and 'yes' in result.lower():
            suspicious_detected = True
            print(f"⚠️ Suspicious activity detected in frame {idx}")
            break
    
    # Stage 2: Detailed analysis if suspicious activity found
    if suspicious_detected:
        print(f"📊 Stage 2: Detailed behavior analysis...")
        
        detailed_prompt = f"""You are an expert security analyst reviewing surveillance footage. A {object_type} has been detected with {confidence:.0%} confidence.

Analyze the provided image and provide a comprehensive security report covering:

1. **THREAT ASSESSMENT**
   - What threatening behavior or activity is visible?
   - How is the {object_type} being used or displayed?
   - Is there immediate danger to people or property?

2. **BEHAVIORAL ANALYSIS**
   - Describe the individual's actions and body language
   - Are they showing aggressive, threatening, or suspicious behavior?
   - What is the context of the situation?

3. **THREAT LEVEL**
   - Rate the threat level: CRITICAL, HIGH, MEDIUM, or LOW
   - Justify your assessment

4. **RECOMMENDED ACTIONS**
   - What immediate actions should security personnel take?
   - Should law enforcement be contacted?
   - Are evacuations or lockdowns necessary?

Provide a clear, professional security analysis in paragraph form."""

        # Analyze the middle frame (most representative)
        middle_frame = frames[len(frames)//2]
        middle_frame_bytes = frame_to_bytes(middle_frame)
        
        detailed_analysis = analyze_frame_with_gemini(middle_frame_bytes, detailed_prompt)
        
        if detailed_analysis:
            print(f"✅ Detailed analysis complete")
            
            return {
                'success': True,
                'suspicious_activity': True,
                'analysis': detailed_analysis,
                'object_type': object_type,
                'confidence': confidence,
                'frame_count': len(frames),
                'frames_analyzed': len(sample_indices) + 1,  # Quick check + detailed
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'suspicious_activity': True,
                'error': 'Detailed analysis failed',
                'analysis': f'Suspicious activity detected with {object_type}, but detailed analysis unavailable.',
                'object_type': object_type,
                'confidence': confidence
            }
    
    else:
        # No suspicious activity detected
        print(f"✅ No suspicious activity detected")
        
        analysis = f"""BEHAVIOR ANALYSIS REPORT

Detected Object: {object_type.upper()}
Detection Confidence: {confidence:.2%}
Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ASSESSMENT: NO IMMEDIATE THREAT DETECTED

After analyzing the surveillance footage, the presence of the {object_type} does not appear to be associated with threatening or suspicious behavior at this time. The object may be:
- Being carried in a non-threatening manner
- Part of normal authorized activities
- Secured or holstered appropriately

THREAT LEVEL: LOW TO MEDIUM

The detection still warrants monitoring as the presence of a {object_type} requires vigilance.

RECOMMENDED ACTIONS:
1. Continue monitoring the individual/area
2. Verify if the person is authorized to carry the {object_type}
3. Keep security personnel informed
4. No immediate intervention required unless behavior changes

NOTE: This is an AI assessment. Human judgment should always be applied for final security decisions."""

        return {
            'success': True,
            'suspicious_activity': False,
            'analysis': analysis,
            'object_type': object_type,
            'confidence': confidence,
            'frame_count': len(frames),
            'frames_analyzed': len(sample_indices),
            'timestamp': datetime.now().isoformat()
        }


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    API endpoint to receive and analyze video frames.
    
    Expected JSON payload:
    {
        "model": "gemini-2.5-flash-lite",
        "frames": ["base64_frame1", "base64_frame2", ...],
        "prompt": "analysis prompt",
        "object_type": "gun",
        "confidence": 0.95,
        "frame_count": 15,
        "fps": 3,
        "duration": 5
    }
    """
    try:
        data = request.json
        
        # Extract parameters
        base64_frames = data.get('frames', [])
        object_type = data.get('object_type', 'unknown')
        confidence = data.get('confidence', 0.0)
        
        print(f"\n{'='*70}")
        print(f"🎬 Received VLM Analysis Request")
        print(f"{'='*70}")
        print(f"Object Type: {object_type}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Frames: {len(base64_frames)}")
        print(f"Model: Gemini 2.5 Flash Lite")
        print(f"{'='*70}\n")
        
        # Convert base64 frames to OpenCV format
        frames = []
        for i, b64_frame in enumerate(base64_frames):
            try:
                frame = base64_to_frame(b64_frame)
                frames.append(frame)
            except Exception as e:
                print(f"⚠️ Error decoding frame {i}: {str(e)}")
        
        if not frames:
            return jsonify({
                'success': False,
                'error': 'No valid frames received',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        print(f"✅ Successfully decoded {len(frames)} frames")
        
        # Analyze with Gemini
        result = analyze_frames_sequence(frames, object_type, confidence)
        
        if result['success']:
            print(f"✅ Analysis complete!")
            if result.get('suspicious_activity'):
                print(f"🚨 SUSPICIOUS ACTIVITY CONFIRMED")
            else:
                print(f"✅ No immediate threat detected")
            print(f"📊 Analysis preview: {result['analysis'][:150]}...\n")
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    gemini_status = "healthy" if client else "unavailable"
    return jsonify({
        'status': 'healthy' if client else 'degraded',
        'service': 'VLM Behavior Analysis API (Gemini 2.5 Flash)',
        'gemini_status': gemini_status,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/', methods=['GET'])
def index():
    """API information endpoint."""
    return jsonify({
        'service': 'VLM Behavior Analysis API',
        'version': '1.0.0',
        'model': 'Gemini 2.5 Flash Lite',
        'project': 'pristine-sphere-477402-t2',
        'endpoints': {
            '/analyze': 'POST - Analyze video frames for suspicious behavior',
            '/health': 'GET - Health check',
            '/': 'GET - API information'
        },
        'status': 'online',
        'gemini_available': client is not None
    }), 200


@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify Gemini integration."""
    if not client:
        return jsonify({
            'status': 'error',
            'message': 'Gemini client not initialized'
        }), 500
    
    try:
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:] = (128, 128, 128)  # Gray image
        test_bytes = frame_to_bytes(test_image)
        
        # Test Gemini
        result = analyze_frame_with_gemini(test_bytes, "Describe this image in one word.")
        
        return jsonify({
            'status': 'success',
            'message': 'Gemini API is working',
            'test_response': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting VLM Behavior Analysis API Server")
    print("   Powered by Google Gemini 2.5 Flash Lite")
    print("="*70)
    print("📡 Server: http://localhost:8000")
    print("🔑 Project: pristine-sphere-477402-t2")
    print("📝 Endpoints:")
    print("   POST /analyze - Analyze video frames")
    print("   GET  /health  - Health check")
    print("   GET  /test    - Test Gemini connection")
    print("   GET  /        - API information")
    print("="*70)
    
    if not client:
        print("\n⚠️  WARNING: Gemini client not initialized!")
        print("   Check your Google Cloud credentials and project ID")
    else:
        print("\n✅ Gemini client ready")
    
    print("="*70 + "\n")
    
    # Run server
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)