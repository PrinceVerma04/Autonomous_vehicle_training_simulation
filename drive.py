import socketio
import eventlet
import numpy as np
from flask import Flask
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
import cv2
import os

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize socketio server
sio = socketio.Server()
app = Flask(__name__)

# Speed limit for throttle control
speed_limit = 10

def img_preprocess(img):
    """Preprocess image - MUST match training preprocessing exactly"""
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    """Receives telemetry data from simulator"""
    try:
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.array([image])
        
        steering_angle = float(model.predict(image, verbose=0))
        throttle = 1.0 - speed / speed_limit
        
        print(f'Steering: {steering_angle:+.4f} | Throttle: {throttle:.4f} | Speed: {speed:5.2f}')
        send_control(steering_angle, throttle)
        
    except Exception as e:
        print(f"Error in telemetry: {e}")

@sio.on('connect')
def connect(sid, environ):
    """Handle connection event"""
    print('✓ Connected to simulator')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    """Send control commands to simulator"""
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    print("=" * 70)
    print("  SELF-DRIVING CAR SERVER - NVIDIA CNN MODEL")
    print("=" * 70)
    
    # Model search paths
    model_paths = [
        'model/model.h5',
        'model/model.keras',
        'model/model_fixed.h5',
        'model/model_fixed.keras',
    ]
    
    model = None
    loaded_path = None
    
    print("\n Searching for model file...")
    print("-" * 70)
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue
            
        file_size = os.path.getsize(model_path) / (1024*1024)
        print(f"\n Found: {model_path} ({file_size:.2f} MB)")
        
        if file_size < 0.5:
            print(f"     File too small - likely corrupted!")
            continue
        
        try:
            print(f"    Loading model...")
            # Load WITHOUT compile to avoid compatibility issues
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Manually recompile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='mse'
            )
            
            loaded_path = model_path
            print(f"    Successfully loaded!")
            break
            
        except Exception as e:
            error_msg = str(e)
            if "batch_shape" in error_msg.lower():
                print(f"    COMPATIBILITY ERROR: batch_shape issue")
                print(f"   Solution: Re-run the fix code in Google Colab")
            elif "signature not found" in error_msg.lower():
                print(f"    FILE CORRUPTED: Invalid HDF5 file")
                print(f"    Solution: Re-download from Colab")
            else:
                print(f"    Error: {error_msg[:60]}...")
            continue
    
    if model is None:
        print("\n" + "=" * 70)
        print(" CRITICAL ERROR: NO VALID MODEL FOUND!")
        print("=" * 70)
        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("\n1. Go back to Google Colab")
        print("2. Add the model fix code at the end of your notebook")
        print("3. Run it to generate 'model_fixed.h5'")
        print("4. Download 'model_fixed.h5'")
        print("5. Copy to: model\\model.h5")
        print("\nCommands:")
        print('  Copy-Item "C:\\Users\\saish\\Downloads\\model_fixed.h5" `')
        print('    -Destination "model\\model.h5" -Force')
        print("\n" + "=" * 70)
        input("\nPress Enter to exit...")
        exit(1)
    
    # Display model info
    print("\n" + "=" * 70)
    print("📊 MODEL INFORMATION")
    print("=" * 70)
    print(f"✓ File: {loaded_path}")
    print(f"✓ Size: {os.path.getsize(loaded_path)/(1024*1024):.2f} MB")
    print(f"✓ Input shape: {model.input_shape}")
    print(f"✓ Output shape: {model.output_shape}")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    # Test prediction
    try:
        test_input = np.random.rand(1, 66, 200, 3).astype(np.float32)
        test_pred = model.predict(test_input, verbose=0)
        print(f"✓ Test prediction: {test_pred[0][0]:.6f} (model working!)")
    except Exception as e:
        print(f"⚠️  Warning: Test prediction failed: {e}")
    
    # Start server
    app = socketio.Middleware(sio, app)
    
    print("\n" + "=" * 70)
    print("🚀 SERVER STATUS")
    print("=" * 70)
    print('✓ Server starting on http://localhost:4567')
    print('✓ Waiting for simulator connection...')
    print("\n📝 NEXT STEPS:")
    print("1. Launch Udacity Self-Driving Car Simulator")
    print("2. Select 'AUTONOMOUS MODE'")
    print("3. Choose 'Lake Track'")
    print("4. Watch your car drive!")
    print("=" * 70 + "\n")
    
    try:
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")
