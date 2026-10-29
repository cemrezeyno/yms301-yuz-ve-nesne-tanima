# yms301-yuz-ve-nesne-tanima
Python ile görüntü işleme tabanlı yüz ve nesne tanıma sistemi

## Teknolojiler
Python 3.10+, Streamlit, OpenCV, YOLO, SQLite, pytest

## Klasör yapısı
```
yms301-yuz-ve-nesne-tanima/
├── main.py                      # Giriş noktası (yalnızca yönlendirme)
├── src/
│   ├── core/
│   │   ├── face_recognition.py  # 1. kişi: yüz tespiti ve kişi eşleştirme
│   │   └── object_detection.py  # 2. kişi: YOLO ile nesne tespiti
│   ├── data/
│   │   └── database.py          # 3. kişi: SQLite veritabanı katmanı
│   └── web/
│       └── app.py               # 3. kişi: Streamlit arayüzü
├── tests/                       # pytest birim testleri
└── data/                        # project.db burada oluşur (Git'e eklenmez)
```

## Veritabanı
| Tablo | İçerik |
|---|---|
| `persons` | İzinli kayıtlı kişiler (ad, kayıt tarihi, aktif/pasif) |
| `analyses` | Her analiz: görsel adı, tür (`face` / `object` / `both`), durum, hata mesajı |
| `face_results` | Bulunan yüzler: kutu konumu, güven skoru, eşleşen kişi (yoksa "Bilinmeyen kişi") |
| `object_results` | Bulunan nesneler: sınıf adı, güven skoru, kutu konumu |

`face_results` ve `object_results` tabloları `analyses` tablosuna bağlıdır. Bir analiz silinince sonuçları da silinir.
Bir kişi silinince eski yüz sonuçları "Bilinmeyen kişi" olarak kalır.

### Modüller arası veri formatı
Yüz modülünün döndürdüğü her sonuç:
```python
{"person_id": 3, "confidence": 0.87, "box": {"x": 10, "y": 20, "width": 50, "height": 60}}
```
Nesne modülünün döndürdüğü her sonuç:
```python
{"object_name": "laptop", "confidence": 0.95, "box": {"x": 10, "y": 20, "width": 50, "height": 60}}
```

## Kurulum
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data.database      # veritabanını oluşturur
```

## Testler
```powershell
pytest -v
```
