import io
import base64

from PIL import Image


def base64_encode_image(image: Image.Image, format: str = "PNG") -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return encoded_image


def image_decode_base64(base64_string: str,) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(base64_string)))


def base64_encode_image_list(image_list: list[Image.Image], format: str = "PNG") -> list[str]:
    return [base64_encode_image(image, format) for image in image_list]


def image_decode_base64_list(base64_string_list: list[str]) -> list[Image.Image]:
    return [image_decode_base64(base64_string) for base64_string in base64_string_list]

