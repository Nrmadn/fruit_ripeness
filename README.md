# Fruit Ripeness Classification: MobileNet Transfer Learning

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.19.0-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-Fruits%20360-yellow.svg)](https://www.kaggle.com/datasets/moltean/fruits)

Sistem klasifikasi kematangan buah menggunakan *pre-trained MobileNet* dengan transfer learning untuk mengklasifikasikan pisang dan mangga ke dalam 6 kategori kematangan, diimplementasikan dalam aplikasi web berbasis Flask.

## 📋 Deskripsi

Penelitian ini mengimplementasikan pre-trained MobileNet model dengan modifikasi arsitektur untuk klasifikasi kematangan dua jenis buah (pisang dan mangga) menjadi enam kategori: Pisang Mentah, Pisang Terlalu Matang, Pisang Matang, Mangga Mentah, Mangga Setengah Matang, dan Mangga Matang. Sistem ini mengatasi keterbatasan metode manual yang subjektif dan tidak konsisten.

### Hasil Utama
- *Akurasi: 98.00%* - Melampaui target penelitian (75-85%) dengan margin +13-23 poin
- *Precision, Recall, F1-Score: 98.00%* - Balance sempurna tanpa bias kelas
- *Training Time: 3.61 jam* untuk 32 epochs dengan early stopping optimal
- *Inference Time: ~52ms* per gambar - Cocok untuk aplikasi real-time
- *Model Size: 13.49 MB* - Lightweight dan deployment-ready

## 🎯 Fitur

### Model & Training
- ✅ Transfer learning dengan pre-trained MobileNet (ImageNet weights)
- ✅ Fine-tuning strategy dengan 84% trainable parameters
- ✅ Data augmentation komprehensif (rotation, zoom, brightness, flip)
- ✅ Adaptive learning rate dengan ReduceLROnPlateau
- ✅ Early stopping untuk efisiensi training
- ✅ Model checkpoint untuk best model selection

### Aplikasi Web
- ✅ Upload gambar (JPG, JPEG, PNG) dengan drag & drop
- ✅ Real-time prediction dengan confidence score
- ✅ Visualisasi hasil dengan bar chart untuk semua kelas
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Processing time display
- ✅ Error handling yang informatif
- ✅ Support hingga 20 concurrent users

### Evaluasi
- ✅ 6 metrik evaluasi (Accuracy, Precision, Recall, F1-Score, RMSE, MAE)
- ✅ Confusion matrix analysis
- ✅ Learning curves visualization
- ✅ Feature importance analysis
- ✅ Perbandingan dengan baseline CNN

## 🗂 Struktur Repository

```bash
fruit_ripeness/
├── README.md                           # Dokumentasi utama
├── requirements.txt                    # Python dependencies
├── app.py                             # Flask web application
├── model/
│   ├── mobilenet_model.h5             # Trained MobileNet model
│   ├── training_notebook.ipynb        # Jupyter notebook training
│   └── model_architecture.png         # Visualisasi arsitektur
├── static/
│   ├── css/
│   │   └── style.css                  # Custom styles
│   ├── js/
│   │   └── script.js                  # Client-side logic
│   ├── uploads/                       # Temporary image storage
│   └── results/                       # Generated visualizations
├── templates/
│   ├── index.html                     # Landing page
│   └── result.html                    # Result display page
├── data/
│   ├── raw/                           # Original Fruits 360 dataset
│   ├── processed/                     # Preprocessed images
│   └── split/                         # Train/val/test splits
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Data preprocessing
│   ├── 03_model_training.ipynb        # Model training & evaluation
│   └── 04_model_comparison.ipynb      # Baseline comparison
├── results/
│   ├── confusion_matrix.png           # Confusion matrix visualization
│   ├── learning_curves.png            # Training/validation curves
│   ├── actual_vs_predicted.png        # Prediction scatter plot
│   ├── feature_importance.png         # Feature importance chart
│   └── model_comparison.csv           # Metrics comparison
├── docs/
│   ├── laporan_penelitian.pdf         # Full research report
│   ├── API.md                         # API documentation
│   └── deployment_guide.md            # Deployment instructions
└── tests/
    ├── test_app.py                    # Flask app tests
    ├── test_model.py                  # Model inference tests
    └── test_preprocessing.py          # Preprocessing tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Git
- CUDA-capable GPU (optional, untuk training)

### Instalasi

1. *Clone repository*
bash
git clone https://github.com/Nrmadn/fruit_ripeness.git
cd fruit_ripeness


2. *Buat virtual environment (recommended)*
bash
python -m venv venv

### Aktivasi di Windows
venv\Scripts\activate

### Aktivasi di Linux/Mac
source venv/bin/activate


3. *Install dependencies*
bash
pip install -r requirements.txt


4. *Download dataset (jika ingin training ulang)*

Dataset Fruits 360 dapat diunduh dari:
- [Kaggle - Fruits 360](https://www.kaggle.com/datasets/moltean/fruits)
- Ekstrak ke folder data/raw/

*Catatan:* Jika hanya ingin menjalankan aplikasi web, trained model sudah tersedia di model/mobilenet_model.h5

5. *Jalankan aplikasi web*
bash
python app.py


Akses aplikasi di: http://localhost:5000

## 📊 Dataset

*Fruits 360* (subset yang digunakan):
- *1,500 gambar* total
- *2 jenis buah*: Pisang dan Mangga
- *6 kategori*: 
  - Banana_Mentah (250 images)
  - Banana_Matang (250 images)
  - Banana_Terlalu_Matang (250 images)
  - Mango_Mentah (250 images)
  - Mango_Setengah_Matang (250 images)
  - Mango_Matang (250 images)
- *Spesifikasi gambar*: 224×224 pixels, RGB
- *Background*: Putih konsisten
- *Split ratio*: 80% training (1,200), 20% validation (300), 20% testing (300)

### Struktur Dataset

```bash
data/
├── raw/
│   ├── Banana/
│   │   ├── Unripe/       # 250 images - Mentah
│   │   ├── Ripe/         # 250 images - Matang
│   │   └── Overripe/     # 250 images - Terlalu Matang
│   └── Mango/
│       ├── Unripe/       # 250 images - Mentah
│       ├── HalfRipe/     # 250 images - Setengah Matang
│       └── Ripe/         # 250 images - Matang
├── processed/
│   └── normalized/       # Preprocessed 224x224 images
└── split/
    ├── train/            # 1,200 images (80%)
    ├── validation/       # 300 images (20%)
    └── test/             # 300 images (20%)
```

## 💻 Usage

### 1. Menjalankan Aplikasi Web

bash
python app.py


Buka browser dan akses: http://localhost:5000

*Workflow:*
1. Upload gambar buah (pisang atau mangga)
2. Preview gambar yang diupload
3. Klik "Prediksi Kematangan"
4. Lihat hasil:
   - Jenis buah
   - Tingkat kematangan
   - Confidence score (%)
   - Distribusi probabilitas semua kelas
   - Processing time

### 2. Training Model dari Awal

python
 Buka Jupyter Notebook
jupyter notebook

 Jalankan notebooks secara berurutan:
 1. notebooks/01_eda.ipynb - Exploratory Data Analysis
 2. notebooks/02_preprocessing.ipynb - Data preparation
 3. notebooks/03_model_training.ipynb - Training & evaluation


Atau jalankan script training:

bash
python train_model.py --epochs 120 --batch_size 32 --learning_rate 0.0002


### 3. Evaluasi Model

python
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

### Load trained model
model = load_model('model/mobilenet_model.h5')

### Load test data (asumsi sudah di-preprocess)
X_test, y_test = load_test_data()

### Prediksi
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

### Evaluasi
print(classification_report(true_classes, predicted_classes))

### Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plot_confusion_matrix(cm)


### 4. Menggunakan Model untuk Prediksi Custom

python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

### Load model
model = load_model('model/mobilenet_model.h5')

### Load dan preprocess gambar
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

### Prediksi
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class] * 100

### Mapping kelas
class_names = [
    'Banana_Matang', 'Banana_Mentah', 'Banana_Terlalu_Matang',
    'Mango_Matang', 'Mango_Mentah', 'Mango_Setengah_Matang'
]

print(f"Predicted: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2f}%")


### 5. API Endpoint (Flask)

python
import requests

### Upload image untuk prediksi
url = 'http://localhost:5000/predict'
files = {'file': open('banana.jpg', 'rb')}
response = requests.post(url, files=files)

result = response.json()
print(f"Fruit Type: {result['fruit_type']}")
print(f"Ripeness: {result['ripeness']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Processing Time: {result['processing_time']:.2f}ms")


## 📈 Hasil Penelitian

### Perbandingan dengan Penelitian Terdahulu

| Penelitian | Jenis Buah | Kelas | Akurasi | Model |
|------------|------------|-------|---------|-------|
| Shahi et al. (2021) | 131 jenis | 131 | 99.92% | MobileNetV2 + Attention |
| Sintiya et al. (2025) | Pisang (1) | 3 | 92.3% | CNN-LSTM |
| Hanifah & Hermawan (2023) | Pisang (1) | 3 | 87.8% | Custom CNN |
| Sutrisna et al. (2024) | Pepaya (1) | 4 | 90.1% | Custom CNN |
| *Penelitian Ini* | *2 jenis* | *6* | *98.00%* | *MobileNet + Fine-tuning* |

### Metrik Evaluasi

| Metrik | Nilai | Keterangan |
|--------|-------|------------|
| *Accuracy* | 98.00% | Proporsi prediksi benar dari total 300 prediksi |
| *Precision* | 98.00% | Ketepatan prediksi positif (macro average) |
| *Recall* | 98.00% | Kemampuan mendeteksi kelas positif (macro avg) |
| *F1-Score* | 98.00% | Harmonic mean precision dan recall |
| *RMSE* | 0.0891 | Root Mean Square Error pada validation |
| *Training Time* | 3.61 hours | 32 epochs dengan early stopping |
| *Inference Time* | 52 ms | Average per image |
| *Model Size* | 13.49 MB | Compact untuk deployment |

### Performa Per Kelas

| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Banana_Matang | 98% | 98% | 98% | 50 |
| Banana_Mentah | 98% | 100% | 99% | 50 |
| Banana_Terlalu_Matang | 100% | 98% | 99% | 50 |
| Mango_Matang | 96% | 96% | 96% | 50 |
| Mango_Mentah | 100% | 100% | 100% | 50 |
| Mango_Setengah_Matang | 96% | 96% | 96% | 50 |
| *Macro Average* | *98%* | *98%* | *98%* | *300* |

### Perbandingan MobileNet vs Baseline CNN

| Aspek | MobileNet (Transfer Learning) | Baseline CNN (From Scratch) | Improvement |
|-------|------------------------------|----------------------------|-------------|
| *Akurasi* | 98.00% | 84.11% | +13.89% |
| *Training Time* | 3.61 hours | 8.82 hours | 59% lebih cepat |
| *Epochs to Converge* | 32 | 87 | 51% lebih sedikit |
| *Inference Time* | 52 ms | 62 ms | 27% lebih cepat |
| *Trainable Params* | 2.97M | 5.8M | 49% lebih efisien |
| *Overfitting Risk* | Low | Medium | ✓ Better |

### Hyperparameter Optimal

*MobileNet Configuration:*
- Base Model: MobileNet (pre-trained ImageNet)
- Frozen Layers: 0-49 (feature extractor)
- Trainable Layers: 50-87 (domain adaptation)
- Optimizer: Adam (initial lr=2×10⁻⁴)
- Loss Function: Categorical Cross-Entropy
- Batch Size: 32
- Early Stopping: Patience 15 epochs
- ReduceLROnPlateau: Patience 7, factor 0.5

*Custom Layers:*
python
GlobalAveragePooling2D()
Dense(256, activation='relu')
BatchNormalization()
Dropout(0.4)
Dense(128, activation='relu')
BatchNormalization()
Dropout(0.3)
Dense(64, activation='relu')
Dropout(0.3)
Dense(6, activation='softmax')  # Output layer


### Confusion Matrix Analysis

```bash
Predicted →
Actual ↓   B_Men  B_Mat  B_TM  M_Men  M_SM  M_Mat
B_Mentah    50     0      0     0      0     0    ✓ Perfect
B_Matang     0    49      1     0      0     0    
B_TM         0     1     49     0      0     0    
M_Mentah     0     0      0    50      0     0    ✓ Perfect
M_SM         0     0      0     0     48     2    
M_Matang     0     0      0     0      2    48    
```

*Key Insights:*
- Error rate hanya *2%* (6 kesalahan dari 300 prediksi)
- Tidak ada misclassification antar jenis buah (Banana vs Mango)
- Kesalahan hanya pada boundary cases:
  - Banana_Matang ↔ Banana_Terlalu_Matang: 1 kesalahan
  - Mango_Setengah_Matang ↔ Mango_Matang: 2 kesalahan

## 🔬 Metodologi

Penelitian menggunakan framework *CRISP-DM* (Cross-Industry Standard Process for Data Mining):

### 1. Business Understanding
- *Problem*: Klasifikasi kematangan buah secara manual subjektif dan tidak konsisten
- *Solution*: Sistem otomatis berbasis deep learning dengan transfer learning
- *Target*: Akurasi ≥75-85% dengan inference time <100ms

### 2. Data Understanding
- *Dataset*: Fruits 360 (1,500 gambar, 6 kelas)
- *EDA Findings*:
  - Distribusi kelas seimbang (250 per kelas)
  - Background konsisten (putih)
  - Pencahayaan standar
  - Variasi warna yang jelas antar kategori

### 3. Data Preparation
- *Preprocessing*:
  - Resize ke 224×224 pixels
  - Normalisasi pixel values ke [0, 1]
  - Label encoding untuk 6 kelas
  - Train-validation-test split (80-20-20)

- *Data Augmentation* (training only):
  - Rotation: ±30°
  - Zoom: ±20%
  - Brightness: 0.7-1.3
  - Horizontal flip
  - Width/Height shift: ±20%

### 4. Modeling
- *Base Model*: Pre-trained MobileNet (ImageNet weights)
- *Strategy*: Transfer learning dengan fine-tuning
- *Architecture Modification*: 
  - Freeze layer 0-49
  - Unfreeze layer 50-87
  - Add custom classification head
- *Optimization*: Grid Search CV untuk hyperparameter

### 5. Evaluation
- *Metrics*: Accuracy, Precision, Recall, F1-Score, RMSE, MAE
- *Validation*: 5-fold cross-validation
- *Comparison*: Baseline CNN (from scratch)
- *Analysis*: Confusion matrix, learning curves, feature importance

### 6. Deployment
- *Platform*: Flask web application
- *Interface*: Responsive HTML/CSS/JS
- *Features*: Upload, predict, visualize results
- *Testing*: Functional testing (10 scenarios, 100% pass rate)

## 📚 Dependencies

### Core Libraries

txt
### Deep Learning
tensorflow==2.19.0
keras==3.7.0

### Web Framework
flask==3.0.0
flask-cors==4.0.0

### Data Processing
numpy>=1.21.0
pandas>=1.3.0
pillow>=9.0.0

### Machine Learning
scikit-learn>=0.24.0
scipy>=1.7.0

### Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

### Utilities
python-dotenv>=0.19.0
werkzeug>=3.0.0


### Instalasi

bash
### Metode 1: Dari requirements.txt
pip install -r requirements.txt

### Metode 2: Install manual
pip install tensorflow keras flask numpy pandas pillow scikit-learn matplotlib seaborn


### Optional (untuk training dengan GPU)

bash
### CUDA Toolkit 11.8
### cuDNN 8.6
pip install tensorflow-gpu==2.19.0


## 🏗 Arsitektur Sistem

### Model Architecture

```bash
Input (224x224x3)
        ↓
┌───────────────────────┐
│   MobileNet Base      │
│   (pre-trained)       │
│   Layer 0-49: Frozen  │
│   Layer 50-87: Train  │
└───────────────────────┘
        ↓
GlobalAveragePooling2D
        ↓
Dense(256) + ReLU
        ↓
BatchNormalization
        ↓
Dropout(0.4)
        ↓
Dense(128) + ReLU
        ↓
BatchNormalization
        ↓
Dropout(0.3)
        ↓
Dense(64) + ReLU
        ↓
Dropout(0.3)
        ↓
Dense(6) + Softmax
        ↓
Output (6 classes)
```

### Web Application Architecture

```bash
┌─────────────────┐
│   Client        │
│   (Browser)     │
└────────┬────────┘
         │ HTTP Request (multipart/form-data)
         ↓
┌─────────────────┐
│   Flask App     │
│   (app.py)      │
├─────────────────┤
│ • Routing       │
│ • File handling │
│ • Preprocessing │
└────────┬────────┘
         │ Preprocessed image (224x224x3)
         ↓
┌─────────────────┐
│   Model         │
│   (mobilenet)   │
├─────────────────┤
│ • Inference     │
│ • Prediction    │
└────────┬────────┘
         │ Prediction + Confidence
         ↓
┌─────────────────┐
│   Response      │
│   (JSON)        │
├─────────────────┤
│ • Fruit type    │
│ • Ripeness      │
│ • Confidence    │
│ • Distribution  │
└─────────────────┘

```
## 🧪 Testing

### Unit Tests

bash
### Jalankan semua tests
pytest tests/

### Test spesifik
pytest tests/test_model.py -v
pytest tests/test_app.py -v
pytest tests/test_preprocessing.py -v


### Functional Testing

Hasil testing aplikasi web (10 skenario):

| No | Skenario | Expected | Actual | Status |
|----|----------|----------|--------|--------|
| 1 | Upload JPG valid | Berhasil | Berhasil | ✅ |
| 2 | Upload PNG valid | Berhasil | Berhasil | ✅ |
| 3 | Upload file invalid (.pdf) | Error | Error | ✅ |
| 4 | Upload file >5MB | Error | Error | ✅ |
| 5 | Prediksi pisang mentah | Mentah >90% | 96.8% | ✅ |
| 6 | Prediksi mangga setengah matang | Setengah Matang | 87.2% | ✅ |
| 7 | Response time | <100ms | 45-60ms | ✅ |
| 8 | Concurrent 20 users | No degradation | Stable | ✅ |
| 9 | Invalid image format | Error message | Error message | ✅ |
| 10 | Empty upload | Error | Error | ✅ |

*Pass Rate: 100%* (10/10)

## 🚀 Deployment

### Local Deployment

bash
### Development mode
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py


### Production Deployment

bash
### Using Gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

### Using Waitress (Windows-friendly)
pip install waitress
waitress-serve --port=8000 app:app


### Docker Deployment

dockerfile
### Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]


bash
### Build dan run
docker build -t fruit-ripeness .
docker run -p 5000:5000 fruit-ripeness


### Cloud Deployment

*Heroku:*
bash
heroku create fruit-ripeness-app
git push heroku main
heroku open


*Google Cloud Platform:*
bash
gcloud app deploy
gcloud app browse


*AWS Elastic Beanstalk:*
bash
eb init -p python-3.8 fruit-ripeness
eb create fruit-ripeness-env
eb open


## 🤝 Contributing

Kontribusi sangat diterima! Berikut panduan kontribusi:

### Langkah Kontribusi

1. *Fork repository ini*
bash
### Klik tombol "Fork" di GitHub


2. *Clone fork Anda*
bash
git clone https://github.com/YOUR_USERNAME/fruit_ripeness.git
cd fruit_ripeness


3. *Buat branch baru*
bash
git checkout -b feature/AmazingFeature


4. *Commit perubahan*
bash
git add .
git commit -m 'Add some AmazingFeature'


5. *Push ke branch*
bash
git push origin feature/AmazingFeature


6. *Buat Pull Request*
- Buka GitHub repository Anda
- Klik "New Pull Request"
- Deskripsikan perubahan Anda

### Coding Standards

- Gunakan PEP 8 untuk Python code style
- Tambahkan docstrings untuk functions dan classes
- Write unit tests untuk fitur baru
- Update dokumentasi jika perlu

### Areas untuk Kontribusi

- 🐛 Bug fixes
- ✨ New features (additional fruit types, defect detection)
- 📝 Documentation improvements
- 🧪 Test coverage
- 🎨 UI/UX enhancements
- 🚀 Performance optimizations
- 🌍 Internationalization (i18n)

## 📝 Citation

Jika menggunakan kode atau penelitian ini, mohon cite:

bibtex
@article{diana2025fruit,
  author = {Diana, Nirma Nur},
  title = {Implementasi Pre-trained Convolutional Neural Network (CNN) Model Klasifikasi Kematangan Buah dalam Aplikasi Web},
  year = {2025},
  institution = {UIN Maulana Malik Ibrahim Malang},
  department = {Teknik Informatika, Fakultas Sains dan Teknologi},
  url = {https://github.com/Nrmadn/fruit_ripeness}
}


## 👤 Author

*Nirma Nur Diana*
- NIM: 230605110147
- Universitas: UIN Maulana Malik Ibrahim Malang
- Jurusan: Teknik Informatika
- Fakultas: Sains dan Teknologi
- Email: 230605110147@student.uin-malang.ac.id
- GitHub: [@Nrmadn](https://github.com/Nrmadn)

## 🙏 Acknowledgments

- *Allah SWT* atas segala berkah dan kemudahan dalam penelitian ini
- *Fakultas Sains dan Teknologi* UIN Maulana Malik Ibrahim Malang
-  Prof. Dr. H.MUHAMMAD FAISAL,S.Kom.
- [GroupLens Research](https://grouplens.org/) untuk inspirasi dokumentasi
- [Kaggle - Fruits 360 Dataset](https://www.kaggle.com/datasets/moltean/fruits) oleh Mihai Oltean
- Komunitas open-source TensorFlow, Keras, dan Flask
- Peneliti-peneliti terdahulu yang menjadi referensi:
  - Shahi et al. (2021) - Attention-based MobileNetV2
  - Sintiya et al. (2025) - CNN-LSTM untuk pisang
  - Naranjo-Torres et al. (2020) - CNN review untuk buah

## 📞 Contact & Support

Untuk pertanyaan, saran, bug reports, atau collaboration:

- *GitHub Issues*: [Create Issue](https://github.com/Nrmadn/fruit_ripeness/issues)
- *Email*: 230605110147@student.uin-malang.ac.id
- *WA*: 085100338312



## 🔮 Future Works

Rencana pengembangan untuk versi mendatang:

### Short-term (v2.0)
- [ ] Tambah jenis buah: salak, rambutan, durian, apel
- [ ] Mobile app (Android/iOS) dengan TensorFlow Lite