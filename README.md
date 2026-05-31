# Bitki Tanıma ve Hastalık Tespiti

Derin öğrenme tabanlı bitki türü tanıma ve bitki hastalığı tespiti projesi.

## Proje Yapısı

```
.
├── src/              # Kaynak kodlar
├── data/             # Veri setleri
├── outputs/          # Eğitilmiş modeller ve grafikler
├── docs/
│   └── images/       # README görselleri
├── config.yaml       # Proje yapılandırması
├── requirements.txt  # Python bağımlılıkları
└── README.md
```

## Kurulum

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

1. Veri setinizi `data/` klasörüne yerleştirin.
2. `config.yaml` dosyasındaki ayarları ihtiyacınıza göre düzenleyin.
3. Eğitim ve değerlendirme scriptlerini `src/` altından çalıştırın.

## Çıktılar

Eğitim sonrası modeller `outputs/models/`, grafikler ise `outputs/plots/` klasörüne kaydedilir.
