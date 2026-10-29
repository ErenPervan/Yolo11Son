# 🕳️ YOLO11-Seg-Pothole: Çukur Tespiti için Özelleştirilmiş Model

Bu repo, çukur (pothole) tespiti ve segmentasyonu için özelleştirilmiş bir YOLO11 modelini içermektedir.

## 🚀 Özellikler

### 1. Dynamic Snake Convolution (DSConv)

- **Neden?** Standard konvolüsyon kare şeklindeki kernel'larla çalışır. Çukurlar ise kıvrımlı ve düzensiz kenarlara sahiptir.
- **Çözüm:** DSConv, kernel yapısını dinamik olarak hedef nesnenin şekline göre hizalar. Bu, segmentasyonun çukurun tam sınırlarına oturmasını sağlar.

### 2. Simple Attention Module (SimAM)

- **Neden?** Asfalt üzerindeki yama veya gölge ile gerçek çukuru ayırt etmek zordur.
- **Çözüm:** Parameter eklemeden çalışan bu dikkat mekanizması, modelin "önemli" piksellere (çukur içi) odaklanmasını sağlar.

### 3. GELU Aktivasyon Fonksiyonu

- **Neden?** Varsayılan SiLU yerine GELU kullanılarak öğrenmenin daha stabil olması sağlanmıştır.

## 📁 Dosya Yapısı

```
ultralytics/
├── nn/
│   └── modules/
│       └── custom.py          # Özel modüller (DSConv, SimAM, ConvGELU, vb.)
├── cfg/
│   └── models/
│       └── 11/
│           ├── yolo11-seg-pothole.yaml       # Ana model konfigürasyonu
│           └── yolo11-seg-pothole-lite.yaml  # Hafif versiyon
└── POTHOLE_README.md          # Bu dosya
```

## 🛠️ Kurulum

### Google Colab İçin

```python
# GitHub'dan repo'yu klonla
!git clone https://github.com/YOUR_USERNAME/yolo11-pothole-detection.git
%cd yolo11-pothole-detection

# Ultralytics'i yükle (düzenlenmiş versiyon)
!pip install -e .
```

### Yerel Kurulum

```bash
git clone https://github.com/YOUR_USERNAME/yolo11-pothole-detection.git
cd yolo11-pothole-detection
pip install -e .
```

## 📊 Model Varyantları

| Model               | Parameter   | Kullanım                     |
| ------------------- | ----------- | ---------------------------- |
| yolo11n-seg-pothole | En az       | Mobil/Edge cihazlar          |
| yolo11s-seg-pothole | Orta        | **Önerilen**                 |
| yolo11m-seg-pothole | Orta-Yüksek | Yüksek doğruluk gerektiğinde |
| yolo11l-seg-pothole | Yüksek      | Maksimum doğruluk            |
| yolo11x-seg-pothole | En yüksek   | Araştırma amaçlı             |

## 🎯 Eğitim

### Google Colab'da Eğitim

```python
from ultralytics import YOLO

# Modeli yükle (sıfırdan)
model = YOLO("ultralytics/cfg/models/11/yolo11-seg-pothole.yaml")

# Veya hafif versiyonu kullan
# model = YOLO('ultralytics/cfg/models/11/yolo11-seg-pothole-lite.yaml')

# Eğitimi başlat
results = model.train(
    data="path/to/pothole-dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    workers=4,
    patience=50,
    save=True,
    project="pothole_detection",
    name="yolo11s-seg-pothole",
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)
```

### Dataset YAML Formatı

```yaml
# pothole-dataset.yaml
path: /content/pothole_dataset # dataset root dir
train: images/train
val: images/val
test: images/test # opsiyonel

# Classes
names:
  0: pothole
```

## 🔍 Tahmin (Inference)

```python
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO("runs/segment/yolo11s-seg-pothole/weights/best.pt")

# Tahmin yap
results = model.predict(
    source="path/to/image.jpg",
    save=True,
    conf=0.25,
    iou=0.45,
    show_labels=True,
    show_conf=True,
    show_boxes=True,
)

# Video üzerinde
results = model.predict(
    source="path/to/video.mp4",
    save=True,
    stream=True,
)
```

## 📈 Değerlendirme

```python
# Validation
metrics = model.val(
    data="path/to/pothole-dataset.yaml",
    batch=16,
    imgsz=640,
    conf=0.25,
    iou=0.6,
)

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Mask mAP50: {metrics.seg.map50}")
print(f"Mask mAP50-95: {metrics.seg.map}")
```

## 🔧 Özel Modüller

### DSConv (Dynamic Snake Convolution)

```python
from ultralytics.nn.modules.custom import DSConv, DySnakeConv

# Tek yönlü DSConv
dsconv = DSConv(in_ch=64, out_ch=128, kernel_size=3, morph=0)  # x-yönü
dsconv = DSConv(in_ch=64, out_ch=128, kernel_size=3, morph=1)  # y-yönü

# Çok yönlü DySnakeConv (x + y + standard)
dysnake = DySnakeConv(c1=64, c2=128, k=3)
```

### SimAM (Simple Attention Module)

```python
from ultralytics.nn.modules.custom import SimAM

# Parametresiz attention
simam = SimAM(e_lambda=1e-4)
output = simam(input_tensor)  # Kanal boyutu korunur
```

### ConvGELU

```python
from ultralytics.nn.modules.custom import ConvGELU

# GELU aktivasyonlu convolution
conv = ConvGELU(c1=64, c2=128, k=3, s=1)
```

## 📝 Notlar

1. **GPU Bellek:** DSConv ve SimAM ekstra bellek kullanır. Batch size'ı buna göre ayarlayın.
2. **Eğitim Süresi:** Özel modüller nedeniyle eğitim biraz daha uzun sürebilir.
3. **Pretrained Weights:** Bu model sıfırdan eğitilmelidir çünkü mimari farklıdır.

## 📚 Referanslar

- [Dynamic Snake Convolution Paper](https://arxiv.org/abs/2307.08388)
- [SimAM Paper](https://proceedings.mlr.press/v139/yang21o)
- [GELU Activation](https://arxiv.org/abs/1606.08415)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## 📄 Lisans

AGPL-3.0 License - Ultralytics lisansı altında dağıtılmaktadır.
