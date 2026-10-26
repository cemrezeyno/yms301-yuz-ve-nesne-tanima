# Yüz Tanıma Yöntemi Araştırması

## Seçilen yöntem
OpenCV YuNet (yüz tespiti) + SFace (yüz tanıma, Hafta 3). Her ikisi de ONNX modelidir.

## Neden bu yöntem?
- Proje zaten OpenCV kullanıyor; TensorFlow veya dlib gibi ağır bağımlılık gerekmiyor.
- Windows'ta kurulum sorunsuz (cmake gerekmez).
- Hızlı çalışıyor, verimlilik kriterine uygun.

## Denemeler ve bulgular
- Python 3.15.1 ile `opencv-python` ve `numpy` sorunsuz kuruldu.
- OpenCV 5.0.0 ve 4.14.0 aynı sonucu verdi. 5.0 ek uyarı mesajı çıkardığı için
  `opencv-python<5` tercih edildi.
- Tam çözünürlüklü görselde (2048 piksel) skor 0.65'te kaldı, yüz bulunamadı.
- Görsel uzun kenarı 320-960 piksele küçültülünce skor 0.92-0.94 oldu.
- Karar: Görsel uzun kenarı ~640 piksele küçültülecek, kutular orijinal boyuta geri ölçeklenecek.

## Alternatifler
- face_recognition (dlib): kurulum zahmetli.
- DeepFace: TensorFlow bağımlılığı ağır.

## Veri güvenliği
Yalnızca izinli fotoğraflar kullanılır. Test fotoğrafları depoya eklenmez.
