"""Streamlit web arayüzü.

İlk ekran: kullanıcı bir görsel yükler, görsel doğrulanır ve önizlenir,
analiz türü seçilir. Yüz ve nesne modülleri hazır olduğunda "Analiz et"
düğmesine bağlanacaktır.

Çalıştırma (proje kök dizininde):
    streamlit run src/web/app.py
"""

import io
import logging
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {"jpg", "jpeg", "png", "bmp", "webp"}
)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Ekranda görünen ad -> veritabanındaki analysis_type değeri
ANALYSIS_OPTIONS: dict[str, str] = {
    "Yüz tanıma": "face",
    "Nesne tanıma": "object",
    "Yüz ve nesne tanıma": "both",
}


class ImageValidationError(Exception):
    """Yüklenen dosya geçerli bir görsel olmadığında fırlatılır.

    Mesajı doğrudan kullanıcıya gösterilecek şekilde yazılır.
    """


# --------------------------------------------------------------------------
# Görsel doğrulama (Streamlit'ten bağımsız, test edilebilir fonksiyonlar)
# --------------------------------------------------------------------------

def validate_extension(filename: str) -> str:
    """Dosya uzantısını kontrol eder ve küçük harfli uzantıyı döndürür."""
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ImageValidationError(
            f"Desteklenmeyen dosya türü: '.{extension or '?'}'. "
            f"İzin verilen türler: {allowed}"
        )
    return extension


def load_image(data: bytes, filename: str) -> Image.Image:
    """Yüklenen baytları doğrular ve RGB bir PIL görseline çevirir.

    Boş, çok büyük, desteklenmeyen uzantılı veya bozuk dosyalarda
    ImageValidationError fırlatır.
    """
    validate_extension(filename)

    if not data:
        raise ImageValidationError("Dosya boş.")
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError(
            f"Dosya çok büyük. En fazla {MAX_FILE_SIZE_MB} MB yüklenebilir."
        )

    try:
        # verify() bozuk dosyaları yakalar ama görseli kullanılamaz bırakır,
        # bu yüzden doğrulamadan sonra görsel yeniden açılır.
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        logger.warning("Bozuk görsel yüklendi: %s (%s)", filename, error)
        raise ImageValidationError(
            "Görsel okunamadı. Dosya bozuk olabilir."
        ) from error

    if image.width == 0 or image.height == 0:
        raise ImageValidationError("Görselin boyutu geçersiz.")

    return image.convert("RGB")


# --------------------------------------------------------------------------
# Ekran parçaları
# --------------------------------------------------------------------------

def render_sidebar() -> str:
    """Kenar çubuğunu çizer ve seçilen analiz türünü döndürür."""
    st.sidebar.header("Ayarlar")
    selected_label = st.sidebar.radio(
        "Analiz türü",
        options=list(ANALYSIS_OPTIONS),
        index=2,
    )
    st.sidebar.caption(
        f"Desteklenen türler: {', '.join(sorted(ALLOWED_EXTENSIONS))}  \n"
        f"En büyük dosya boyutu: {MAX_FILE_SIZE_MB} MB"
    )
    return ANALYSIS_OPTIONS[selected_label]


def render_analysis_page(analysis_type: str) -> None:
    """Görsel yükleme, önizleme ve analiz düğmesini çizer."""
    uploaded_file = st.file_uploader(
        "Analiz edilecek görseli seçin",
        type=sorted(ALLOWED_EXTENSIONS),
    )
    if uploaded_file is None:
        st.info("Başlamak için bir görsel yükleyin.")
        return

    try:
        image = load_image(uploaded_file.getvalue(), uploaded_file.name)
    except ImageValidationError as error:
        st.error(str(error))
        return

    st.image(image, caption=uploaded_file.name, use_container_width=True)
    st.caption(
        f"Boyut: {image.width} × {image.height} piksel  •  "
        f"Dosya: {uploaded_file.size / 1024:.0f} KB"
    )

    if st.button("Analiz et", type="primary"):
        logger.info(
            "Analiz istendi: %s (%s)", uploaded_file.name, analysis_type
        )
        # TODO (5. hafta): yüz ve nesne modüllerini burada çağır,
        # sonuçları çiz ve database.save_analysis() ile kaydet.
        st.warning(
            "Yüz ve nesne modülleri henüz bağlanmadı. "
            "Analiz sonuçları ilerleyen haftalarda burada görünecek."
        )


def main() -> None:
    """Uygulamanın giriş noktası."""
    st.set_page_config(page_title="Yüz ve Nesne Tanıma", layout="centered")
    st.title("Yüz ve Nesne Tanıma Sistemi")
    st.write(
        "Bir görsel yükleyin ve analiz türünü seçin. "
        "Sistem görseldeki yüzleri ve nesneleri tespit eder."
    )

    analysis_type = render_sidebar()
    render_analysis_page(analysis_type)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
