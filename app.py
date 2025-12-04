# ============================================================
# FRUIT RIPENESS CLASSIFICATION - FLASK WEB APPLICATION
# ============================================================
# Author: Nirma Nur Diana
# Model: MobileNet Fine-tuned v3
# Accuracy: 98.00%
# Classes: 6 (Banana: Mentah, Matang, Terlalu Matang | Mango: Mentah, Setengah Matang, Matang)
# ============================================================

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
import time
import h5py

# ============================================================
# KONFIGURASI FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Konfigurasi folder dan file
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Maximum 5MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Secret key untuk session
app.config['SECRET_KEY'] = 'fruit-classifier-secret-key-2025'

# Buat folder uploads jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("="*60)
print("🍌🥭 FRUIT RIPENESS CLASSIFIER - WEB APPLICATION")
print("="*60)

# ============================================================
# KONFIGURASI KELAS (6 CLASSES)
# ============================================================

CLASS_NAMES = [
    'Banana_Matang',
    'Banana_Mentah', 
    'Banana_Terlalu_Matang',
    'Mango_Matang',
    'Mango_Mentah',
    'Mango_Setengah_Matang'
]

num_classes = len(CLASS_NAMES)

# ============================================================
# FUNCTION TO BUILD MODEL (SAMA SEPERTI TRAINING)
# ============================================================

def build_model():
    """Build model architecture - EXACTLY as in training script"""
    print("\n🔨 Building model architecture...")
    
    # Base model
    base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze/unfreeze layers
    for layer in base.layers[:50]:
        layer.trainable = False
    for layer in base.layers[50:]:
        layer.trainable = True
    
    # Build sequential model
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    print(f"✅ Architecture built: {len(model.layers)} layers")
    return model

# ============================================================
# LOAD MODEL - MULTIPLE METHODS
# ============================================================

model = None
model_loaded_from = None

# Method 1: Try loading pre-saved models
MODEL_FILES = [
    'model/best_finetune_v3.h5',
    'model/best_model.keras',
    'model/best_finetune_v3_old.h5',
    'model/BACKUP_old_model.h5',
]

print("\n🔍 Method 1: Trying to load saved models...")

for model_path in MODEL_FILES:
    if os.path.exists(model_path):
        print(f"\n📦 Attempting: {model_path}")
        try:
            # Try with compile=False and safe_mode=False
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            model_loaded_from = model_path
            print(f"✅ SUCCESS!")
            break
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:80]}...")
            
            # Try to extract weights and rebuild
            try:
                print(f"   🔄 Trying to rebuild and load weights...")
                model = build_model()
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                model_loaded_from = f"{model_path} (weights only)"
                print(f"✅ SUCCESS with rebuild method!")
                break
            except Exception as e2:
                print(f"   ❌ Rebuild failed: {str(e2)[:80]}...")
                model = None
                continue

# Method 2: Build fresh model and load weights file
if model is None:
    print("\n🔍 Method 2: Building fresh model and loading weights...")
    WEIGHTS_FILE = 'model/mobilenet.weights.h5'
    
    if os.path.exists(WEIGHTS_FILE):
        print(f"📦 Found weights file: {WEIGHTS_FILE}")
        try:
            model = build_model()
            
            # Try to inspect H5 file structure
            print("   🔍 Inspecting weights file structure...")
            with h5py.File(WEIGHTS_FILE, 'r') as f:
                print(f"   Keys in file: {list(f.keys())}")
            
            # Try loading weights
            print("   🔄 Loading weights...")
            model.load_weights(WEIGHTS_FILE, by_name=True, skip_mismatch=True)
            model_loaded_from = WEIGHTS_FILE
            print(f"✅ SUCCESS! Model built and weights loaded")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            model = None

# Method 3: Extract weights from any .h5 file manually
if model is None:
    print("\n🔍 Method 3: Manual weight extraction from H5 files...")
    
    for model_path in MODEL_FILES:
        if model_path.endswith('.h5') and os.path.exists(model_path):
            print(f"\n📦 Inspecting: {model_path}")
            try:
                # Build fresh model
                model = build_model()
                
                # Try to load weights with skip_mismatch
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                model_loaded_from = f"{model_path} (extracted weights)"
                print(f"✅ SUCCESS! Weights extracted and loaded")
                break
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:80]}...")
                model = None
                continue

# Final check
if model is None:
    print("\n" + "="*60)
    print("⚠️  CRITICAL: NO MODEL LOADED!")
    print("="*60)
    print("\n💡 SOLUTION: Please run this in Google Colab:")
    print("""
# After training, save in multiple formats:
model.save('best_finetune_v3_full.h5')  # Full model
model.save('saved_model_format')         # SavedModel format
model.save_weights('weights_only.h5')    # Weights only

# Then download to your local machine
    """)
    print("="*60)
else:
    print(f"\n✅ MODEL LOADED SUCCESSFULLY!")
    print(f"   Source: {model_loaded_from}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Total params: {model.count_params():,}")
    
    # Compile model
    try:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"   ✅ Model compiled")
    except:
        print(f"   ℹ️  Model already compiled")

CLASS_DISPLAY = {
    'Banana_Matang': {
        'fruit': 'Pisang',
        'fruit_en': 'Banana', 
        'ripeness': 'Matang',
        'ripeness_en': 'Ripe',
        'color': 'warning',
        'emoji': '🍌'
    },
    'Banana_Mentah': {
        'fruit': 'Pisang',
        'fruit_en': 'Banana',
        'ripeness': 'Mentah',
        'ripeness_en': 'Unripe',
        'color': 'success',
        'emoji': '🍌'
    },
    'Banana_Terlalu_Matang': {
        'fruit': 'Pisang',
        'fruit_en': 'Banana',
        'ripeness': 'Terlalu Matang',
        'ripeness_en': 'Overripe',
        'color': 'danger',
        'emoji': '🍌'
    },
    'Mango_Matang': {
        'fruit': 'Mangga',
        'fruit_en': 'Mango',
        'ripeness': 'Matang',
        'ripeness_en': 'Ripe',
        'color': 'warning',
        'emoji': '🥭'
    },
    'Mango_Mentah': {
        'fruit': 'Mangga',
        'fruit_en': 'Mango',
        'ripeness': 'Mentah',
        'ripeness_en': 'Unripe',
        'color': 'success',
        'emoji': '🥭'
    },
    'Mango_Setengah_Matang': {
        'fruit': 'Mangga',
        'fruit_en': 'Mango',
        'ripeness': 'Setengah Matang',
        'ripeness_en': 'Half-ripe',
        'color': 'info',
        'emoji': '🥭'
    }
}

print(f"✅ Loaded {len(CLASS_NAMES)} classes")
print("="*60)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path):
    """Preprocess image untuk prediksi"""
    try:
        img = Image.open(image_path)
        
        # Convert ke RGB jika perlu
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize ke 224x224
        img = img.resize((224, 224), Image.LANCZOS)
        
        # Convert ke array dan normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize ke [0,1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None


def format_confidence(confidence):
    """Format confidence percentage dengan emoji"""
    percentage = confidence * 100
    
    if percentage >= 95:
        return f"🎯 {percentage:.2f}% (Sangat Yakin)"
    elif percentage >= 85:
        return f"✅ {percentage:.2f}% (Yakin)"
    elif percentage >= 70:
        return f"⚠️ {percentage:.2f}% (Cukup Yakin)"
    else:
        return f"❓ {percentage:.2f}% (Kurang Yakin)"

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi gambar"""
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'message': 'Model belum dimuat. Silakan restart aplikasi atau hubungi admin.'
        }), 503
    
    start_time = time.time()
    
    try:
        # Validasi file upload
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded',
                'message': 'Silakan upload gambar terlebih dahulu'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'message': 'Tidak ada file yang dipilih'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type',
                'message': 'Format file tidak valid. Gunakan JPG, JPEG, atau PNG'
            }), 400
        
        # Save file dengan timestamp
        filename = secure_filename(file.filename)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        print(f"📁 File saved: {filepath}")
        
        # Preprocess image
        img_array = preprocess_image(filepath)
        
        if img_array is None:
            return jsonify({
                'success': False,
                'error': 'Image preprocessing failed',
                'message': 'Gagal memproses gambar. Pastikan file adalah gambar yang valid'
            }), 500
        
        print(f"✅ Image preprocessed: shape {img_array.shape}")
        
        # Prediksi
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get hasil prediksi
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        print(f"🎯 Prediction: {predicted_class} ({confidence*100:.2f}%)")
        
        # Get class info
        class_info = CLASS_DISPLAY[predicted_class]
        
        # Semua probabilitas
        all_predictions = {}
        for i, class_name in enumerate(CLASS_NAMES):
            all_predictions[class_name] = {
                'probability': float(predictions[i] * 100),
                'display_name': CLASS_DISPLAY[class_name]['ripeness']
            }
        
        # Hitung inference time
        inference_time = (time.time() - start_time) * 1000
        
        # Response
        response = {
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2),
            'confidence_formatted': format_confidence(confidence),
            'fruit_type': class_info['fruit'],
            'fruit_type_en': class_info['fruit_en'],
            'ripeness': class_info['ripeness'],
            'ripeness_en': class_info['ripeness_en'],
            'emoji': class_info['emoji'],
            'color': class_info['color'],
            'all_predictions': all_predictions,
            'image_path': filepath,
            'inference_time_ms': round(inference_time, 2),
            'message': f"Terdeteksi {class_info['emoji']} {class_info['fruit']} {class_info['ripeness']}",
            'model_source': model_loaded_from
        }
        
        print(f"⏱️ Inference time: {inference_time:.2f}ms")
        print("="*60)
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'message': f'Terjadi kesalahan saat prediksi: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'model_source': model_loaded_from if model is not None else None,
        'classes': len(CLASS_NAMES),
        'version': '1.0.0'
    }), 200 if model is not None else 503


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handler untuk file terlalu besar"""
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': 'Ukuran file terlalu besar! Maksimal 5MB'
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handler untuk internal server error"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'Terjadi kesalahan pada server'
    }), 500


if __name__ == '__main__':
    if model is None:
        print("\n⚠️  WARNING: Server starting WITHOUT model!")
        print("Application will not work until model is loaded properly.\n")
    
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER")
    print("="*60)
    print("📍 Access application at: http://localhost:5000")
    print("📍 Health check at: http://localhost:5000/health")
    if model is not None:
        print(f"📍 Model loaded from: {model_loaded_from}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)