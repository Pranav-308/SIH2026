import io
import pytest
from PIL import Image, ImageDraw

from image_processing import ImageProcessingModule, process_product_image


def create_dummy_image_bytes(width: int = 100, height: int = 100, color: str = "red", format: str = "JPEG") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_process_image_from_bytes():
    image_bytes = create_dummy_image_bytes(200, 200, "green", "JPEG")
    processor = ImageProcessingModule()
    enhanced_img = processor.process_image(image_bytes)

    assert isinstance(enhanced_img, Image.Image)
    assert enhanced_img.mode == "RGB"
    assert enhanced_img.size == (200, 200)


def test_process_image_from_pil_object():
    pil_img = Image.new("RGBA", (150, 150), color=(255, 0, 0, 128))
    processor = ImageProcessingModule()
    enhanced_img = processor.process_image(pil_img)

    assert isinstance(enhanced_img, Image.Image)
    assert enhanced_img.mode == "RGB"  # Converted from RGBA to RGB
    assert enhanced_img.size == (150, 150)


def test_process_image_resize_down():
    # Image exceeding max_dimension (1600)
    image_bytes = create_dummy_image_bytes(2400, 1200, "blue", "JPEG")
    processor = ImageProcessingModule(max_dimension=1600)
    enhanced_img = processor.process_image(image_bytes)

    assert isinstance(enhanced_img, Image.Image)
    assert max(enhanced_img.width, enhanced_img.height) <= 1600
    # Aspect ratio preserved (2:1 ratio => 1600 x 800)
    assert enhanced_img.size == (1600, 800)


def test_process_image_invalid_bytes():
    processor = ImageProcessingModule()
    with pytest.raises(ValueError, match="Image bytes cannot be empty"):
        processor.process_image(b"")

    with pytest.raises(ValueError, match="Invalid image format"):
        processor.process_image(b"not_an_image_stream")


def test_process_image_invalid_type():
    processor = ImageProcessingModule()
    with pytest.raises(TypeError, match="Expected image_input to be bytes or PIL.Image.Image"):
        processor.process_image(12345)


def test_process_image_to_bytes_helper():
    image_bytes = create_dummy_image_bytes(100, 100, "yellow", "PNG")
    processor = ImageProcessingModule()
    jpeg_bytes = processor.process_image_to_bytes(image_bytes, format="JPEG", quality=85)

    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 0
    # Verify it can be opened back as JPEG
    result_img = Image.open(io.BytesIO(jpeg_bytes))
    assert result_img.format == "JPEG"


def test_convenience_function():
    image_bytes = create_dummy_image_bytes(80, 80, "purple", "JPEG")
    enhanced = process_product_image(image_bytes)
    assert isinstance(enhanced, Image.Image)
    assert enhanced.size == (80, 80)


def test_remove_background():
    processor = ImageProcessingModule()
    img = Image.new("RGBA", (100, 100), color=(0, 255, 0, 255))
    clean_bg = processor.remove_background(img)
    assert isinstance(clean_bg, Image.Image)
    assert clean_bg.mode in ("RGB", "RGBA")

