# yms301-yuz-ve-nesne-tanima
Bu proje, Python ve görüntü işleme teknikleri kullanılarak geliştirilen bir yüz ve nesne tanıma sistemidir.

## Proje Hakkında
Sistem, kullanıcı tarafından verilen görüntüleri analiz ederek görüntü içerisindeki yüzleri ve belirlenen nesneleri tespit etmeyi ve tanımayı amaçlamaktadır. Proje kapsamında görüntü işleme, yüz tanıma ve nesne tespiti yöntemleri kullanılacaktır.

## Kullanılacak Teknolojiler
- Python
- OpenCV
- Görüntü işleme

## YOLO Nesne Tespiti Modülü

Proje kapsamında geliştirilen YOLO modülü, verilen görüntülerdeki nesneleri tespit eder ve her tespit için aşağıdaki bilgileri döndürür:

Nesne etiketi: Tespit edilen nesnenin sınıfı.

Güven skoru: Modelin tespit sonucuna ilişkin güven değeri.

Koordinatlar: Tespit edilen nesnenin sınırlayıcı kutusunun x1, y1, x2, y2 koordinatları.

Modül dosyası:

src/core/object_detection.py

## Kurulum

Projeyi bilgisayarınıza indirdikten sonra proje ana klasöründe bir Python sanal ortamı oluşturun ve etkinleştirin.

Windows PowerShell:

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

YOLO model dosyası yolo26n.pt, modülün beklediği şekilde proje ana klasöründe bulunmalıdır. Model dosyası GitHub deposuna eklenmez; ilk kurulumda ayrıca edinilmesi gerekir.

## Kullanım

Proje ana klasöründe aşağıdaki komutla nesne tespiti fonksiyonu çalıştırılabilir:

python -c "from src.core.object_detection import detect_objects; print(detect_objects('bus.jpg'))"

bus.jpg yerine analiz edilecek görüntünün dosya yolu kullanılabilir.

Fonksiyon, her tespit için label, confidence ve box alanlarını içeren bir liste döndürür. Belirtilen görüntü bulunamazsa FileNotFoundError hatası oluşturur.

## Testler

Otomatik testleri çalıştırmak için önce test aracını kurun:

python -m pip install pytest

Ardından:

python -m pytest tests/test_object_detection.py -v

YOLO modülü için iki otomatik test uygulanmıştır:

Geçerli bir görüntüden nesne tespit sonuçlarının döndürülmesi.

Görüntü bulunamadığında uygun hatanın oluşturulması.

Son test çalıştırmasında iki test de başarılı olmuştur.

## Proje Ekibi
- Zeynep Cemre KELEŞ
- Melike KAYA
- Nazar DEMİR

