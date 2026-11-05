"""src/web/app.py için testler.

Görsel doğrulama fonksiyonları doğrudan, ekranın kendisi Streamlit'in
AppTest aracıyla (tarayıcı açmadan) test edilir.
"""

import io
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.web import app

APP_FILE = str(Path(app.__file__).resolve())


def _image_bytes(image_format: str = "PNG", size: tuple[int, int] = (40, 30),
                 mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=0).save(buffer, format=image_format)
    return buffer.getvalue()


# ---------------------------- Uzantı kontrolü -----------------------------

@pytest.mark.parametrize(
    "filename, expected",
    [("foto.jpg", "jpg"), ("FOTO.JPEG", "jpeg"), ("a.b.png", "png")],
)
def test_validate_extension_accepts_images(filename: str,
                                           expected: str) -> None:
    assert app.validate_extension(filename) == expected


@pytest.mark.parametrize("filename", ["belge.pdf", "virus.exe", "dosya"])
def test_validate_extension_rejects_others(filename: str) -> None:
    with pytest.raises(app.ImageValidationError):
        app.validate_extension(filename)


# ---------------------------- Görsel yükleme ------------------------------

def test_load_image_returns_rgb_image() -> None:
    image = app.load_image(_image_bytes(), "foto.png")
    assert image.size == (40, 30)
    assert image.mode == "RGB"


def test_load_image_converts_rgba_and_grayscale() -> None:
    rgba = app.load_image(_image_bytes(mode="RGBA"), "a.png")
    gray = app.load_image(_image_bytes(mode="L"), "b.png")
    assert rgba.mode == "RGB" and gray.mode == "RGB"


def test_load_image_accepts_jpeg() -> None:
    image = app.load_image(_image_bytes("JPEG"), "foto.jpg")
    assert image.size == (40, 30)


def test_load_image_rejects_empty_file() -> None:
    with pytest.raises(app.ImageValidationError, match="boş"):
        app.load_image(b"", "foto.png")


def test_load_image_rejects_corrupt_file() -> None:
    with pytest.raises(app.ImageValidationError, match="okunamadı"):
        app.load_image(b"bu bir resim degil", "foto.png")


def test_load_image_rejects_truncated_file() -> None:
    data = _image_bytes("PNG", size=(200, 200))
    with pytest.raises(app.ImageValidationError):
        app.load_image(data[: len(data) // 2], "yarim.png")


def test_load_image_rejects_wrong_extension() -> None:
    with pytest.raises(app.ImageValidationError, match="Desteklenmeyen"):
        app.load_image(_image_bytes(), "foto.gif")


def test_load_image_rejects_too_large_file(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "MAX_FILE_SIZE_BYTES", 10)
    with pytest.raises(app.ImageValidationError, match="büyük"):
        app.load_image(_image_bytes(), "foto.png")


# ---------------------------- Ekran ---------------------------------------

def test_app_starts_without_error() -> None:
    at = AppTest.from_file(APP_FILE).run()
    assert not at.exception
    assert at.title[0].value == "Yüz ve Nesne Tanıma Sistemi"


def test_app_offers_three_analysis_types() -> None:
    at = AppTest.from_file(APP_FILE).run()
    radio = at.sidebar.radio[0]
    assert list(radio.options) == list(app.ANALYSIS_OPTIONS)
    assert radio.value == "Yüz ve nesne tanıma"


def test_app_asks_for_upload_first() -> None:
    at = AppTest.from_file(APP_FILE).run()
    assert "görsel yükleyin" in at.info[0].value
    assert len(at.button) == 0  # görsel yokken "Analiz et" görünmez


def test_changing_analysis_type_keeps_app_running() -> None:
    at = AppTest.from_file(APP_FILE).run()
    at.sidebar.radio[0].set_value("Nesne tanıma").run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Nesne tanıma"
