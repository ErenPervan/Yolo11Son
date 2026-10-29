"""
Google Colab Training Script for YOLO11-Seg-Pothole.
====================================================

Bu script'i Google Colab'da çalıştırarak çukur tespiti modelinizi eğitebilirsiniz.

Kullanım:
1. Bu repo'yu GitHub'a push edin
2. Google Colab'da yeni bir notebook açın
3. Aşağıdaki hücreleri sırasıyla çalıştırın
"""

# ============================================================================
# CELL 1: GPU Kontrolü ve Kurulum
# ============================================================================
"""
# GPU kontrol
!nvidia-smi

# Repo'yu klonla (YOUR_USERNAME'i kendi GitHub kullanıcı adınızla değiştirin)
!git clone https://github.com/YOUR_USERNAME/yolo11-pothole-detection.git
%cd yolo11-pothole-detection

# Ultralytics'i düzenlenmiş versiyondan yükle
!pip install -e . -q

# Test modülleri
!python test_pothole_model.py
"""

# ============================================================================
# CELL 2: Dataset Hazırlama
# ============================================================================
"""
# Örnek: Roboflow'dan dataset çekme
# Not: Kendi dataset'inizi kullanacaksanız bu adımı atlayın

# Roboflow kurulumu
!pip install roboflow -q

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov8")

# Dataset path'i kontrol
!ls -la {dataset.location}
"""

# ============================================================================
# CELL 3: Dataset YAML Oluşturma
# ============================================================================
"""
# Eğer kendi dataset'inizi yüklediyseniz, yaml dosyasını oluşturun:

yaml_content = '''
path: /content/yolo11-pothole-detection/dataset
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: pothole
'''

with open('pothole_dataset.yaml', 'w') as f:
    f.write(yaml_content)
    
print("Dataset YAML oluşturuldu!")
"""

# ============================================================================
# CELL 4: Model Eğitimi
# ============================================================================
"""
from ultralytics import YOLO

# Model seçenekleri:
# 1. Ana versiyon (daha fazla DSConv ve SimAM)
# model = YOLO('ultralytics/cfg/models/11/yolo11-seg-pothole.yaml')

# 2. Lite versiyon (daha hızlı, daha az kaynak)
model = YOLO('ultralytics/cfg/models/11/yolo11-seg-pothole-lite.yaml')

# Model bilgisi
print(model.info())

# Eğitimi başlat
results = model.train(
    data='pothole_dataset.yaml',  # Dataset yaml path
    epochs=100,                    # Epoch sayısı
    imgsz=640,                     # Görüntü boyutu
    batch=16,                      # Batch size (GPU belleğine göre ayarla)
    device=0,                      # GPU
    workers=4,                     # Dataloader workers
    patience=50,                   # Early stopping patience
    save=True,                     # Checkpoint kaydet
    save_period=10,                # Her 10 epoch'ta kaydet
    project='pothole_detection',  # Proje adı
    name='yolo11s-seg-pothole',   # Çalışma adı
    exist_ok=True,                 # Varsa üzerine yaz
    
    # Pretrained kullanma (çünkü mimari farklı)
    pretrained=False,
    
    # Optimizer
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # Data augmentation
    hsv_h=0.015,                   # Hue augmentation
    hsv_s=0.7,                     # Saturation augmentation
    hsv_v=0.4,                     # Value augmentation
    degrees=10.0,                  # Rotation augmentation
    translate=0.1,                 # Translation augmentation
    scale=0.5,                     # Scale augmentation
    shear=2.0,                     # Shear augmentation
    perspective=0.0,               # Perspective augmentation
    flipud=0.0,                    # Vertical flip
    fliplr=0.5,                    # Horizontal flip
    mosaic=1.0,                    # Mosaic augmentation
    mixup=0.1,                     # Mixup augmentation
    copy_paste=0.1,                # Copy-paste augmentation (segmentation için)
    
    # Loss weights
    box=7.5,                       # Box loss weight
    cls=0.5,                       # Classification loss weight
    dfl=1.5,                       # DFL loss weight
    
    # Diğer
    amp=True,                      # Automatic Mixed Precision
    verbose=True,
    plots=True,                    # Eğitim grafikleri
)

print("Eğitim tamamlandı!")
"""

# ============================================================================
# CELL 5: Model Değerlendirme
# ============================================================================
"""
# Eğitilmiş modeli yükle
model = YOLO('pothole_detection/yolo11s-seg-pothole/weights/best.pt')

# Validation
metrics = model.val(
    data='pothole_dataset.yaml',
    batch=16,
    imgsz=640,
    conf=0.25,
    iou=0.6,
    device=0,
)

# Sonuçları yazdır
print(f"\\n{'='*50}")
print("Model Değerlendirme Sonuçları")
print('='*50)
print(f"Box mAP50:     {metrics.box.map50:.4f}")
print(f"Box mAP50-95:  {metrics.box.map:.4f}")
print(f"Mask mAP50:    {metrics.seg.map50:.4f}")
print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
print(f"Precision:     {metrics.box.mp:.4f}")
print(f"Recall:        {metrics.box.mr:.4f}")
print('='*50)
"""

# ============================================================================
# CELL 6: Tahmin (Inference)
# ============================================================================
"""
from IPython.display import Image, display
import glob

# Test görüntüleri üzerinde tahmin
model = YOLO('pothole_detection/yolo11s-seg-pothole/weights/best.pt')

# Tek görüntü tahmini
results = model.predict(
    source='path/to/test/image.jpg',
    save=True,
    conf=0.25,
    iou=0.45,
    show_labels=True,
    show_conf=True,
    show_boxes=True,
    line_width=2,
)

# Sonucu göster
for result in results:
    display(Image(result.path))
"""

# ============================================================================
# CELL 7: Model Export
# ============================================================================
"""
# ONNX formatına export
model = YOLO('pothole_detection/yolo11s-seg-pothole/weights/best.pt')

# ONNX export
model.export(format='onnx', imgsz=640, simplify=True)

# TensorRT export (NVIDIA GPU gerekli)
# model.export(format='engine', imgsz=640)

# TFLite export (mobil cihazlar için)
# model.export(format='tflite', imgsz=640)

print("Export tamamlandı!")
"""

# ============================================================================
# CELL 8: Model İndirme
# ============================================================================
"""
from google.colab import files
import shutil

# En iyi modeli zip'le ve indir
shutil.make_archive('pothole_model', 'zip', 'pothole_detection/yolo11s-seg-pothole/weights')
files.download('pothole_model.zip')

# Tüm sonuçları indir
shutil.make_archive('pothole_results', 'zip', 'pothole_detection/yolo11s-seg-pothole')
files.download('pothole_results.zip')
"""

# ============================================================================
# BONUS: Realtime Video İşleme
# ============================================================================
"""
# Video üzerinde realtime tahmin
model = YOLO('pothole_detection/yolo11s-seg-pothole/weights/best.pt')

results = model.predict(
    source='path/to/video.mp4',
    save=True,
    stream=True,
    conf=0.25,
    iou=0.45,
    show_labels=True,
    show_conf=True,
)

# Stream modunda sonuçları işle
for result in results:
    boxes = result.boxes
    masks = result.masks
    if boxes is not None:
        print(f"Frame: {len(boxes)} pothole detected")
"""

if __name__ == "__main__":
    print("Bu script Google Colab'da çalıştırılmak üzere tasarlanmıştır.")
    print("Her hücreyi ayrı ayrı bir Colab notebook'una kopyalayın.")
    print("\nDetaylı bilgi için POTHOLE_README.md dosyasına bakın.")
