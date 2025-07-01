# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:01:10 2025

@author: darek
"""

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Załaduj model i procesor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Wczytaj obraz
image = Image.open('ścieżka/do/obrazu.png').convert("RGB")

# Przetwórz obraz
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)


from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained(
    'microsoft/trocr-base-handwritten',
    torch_dtype=torch.float16  # FP16, żeby zmniejszyć zużycie VRAM
).to('cuda')

image = Image.open(r"C:/Users/darek/Downloads/Example 1/IMG_7368.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to('cuda')
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Rozpoznany tekst:\n", text)


import os
image_path = r"C:\Users\darek\Downloads\Example 1\IMG_7368.png"
print("Czy plik istnieje?", os.path.exists(image_path))


import os
import imghdr

image_path = r"C:\Users\darek\Downloads\Example 1\IMG_7368.png"

# 1. Sprawdź rozmiar pliku
size = os.path.getsize(image_path)
print(f"Rozmiar pliku: {size/1024:.1f} KB")

# 2. Co wykrywa imghdr?
fmt = imghdr.what(image_path)
print("imghdr rozpoznaje format jako:", fmt)

# 3. Podejrzyj pierwsze 16 bajtów (nagłówek)
with open(image_path, "rb") as f:
    header = f.read(16)
print("Pierwsze 16 bajtów:", header)


C:/Users/darek/Downloads/Example 1/IMG_7368.png

# 0. Instalacja (tylko raz w terminalu):
# pip install pillow-heif transformers torch torchvision accelerate opencv-python-headless

import pillow_heif
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pathlib import Path

def preprocess_pil(image: Image.Image) -> Image.Image:
    # 1) Konwersja na szarość
    gray = image.convert("L")
    np_img = np.array(gray)
    # 2) Progowanie Otsu
    _, thresh = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 3) Deskew
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = thresh.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # 4) Z powrotem do RGB
    return Image.fromarray(rotated).convert("RGB")

# Ścieżka do obrazka
image_path = Path(r"C:\Users\darek\Downloads\Example 1\IMG_7368.png")
if not image_path.exists():
    raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

# 1. Odczyt HEIC → PIL
heif_file = pillow_heif.read_heif(str(image_path))
img = Image.frombytes(
    heif_file.mode,
    heif_file.size,
    heif_file.data,
    "raw",
    heif_file.mode,
    heif_file.stride,
)

# 2. Pre-processing
img_pp = preprocess_pil(img)

# 3. Załaduj model do odręcznego pisma
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten",
    torch_dtype=torch.float16
).to("cuda")  # lub .to("cpu") jeśli nie masz GPU

# 4. Przygotowanie i inferencja
pixel_values = processor(images=img_pp, return_tensors="pt").pixel_values.to(model.device)
generated_ids = model.generate(
    pixel_values,
    max_length=512,
    num_beams=5,
    early_stopping=True
)

# 5. Dekodowanie i wynik
recognized = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Rozpoznany tekst:\n", recognized.strip())


# 0. Instalacja (jednorazowo w terminalu):
# pip install pillow-heif transformers torch torchvision accelerate opencv-python-headless

import pillow_heif
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pathlib import Path

def preprocess_pil(image: Image.Image) -> Image.Image:
    # 1) Konwersja na skalę szarości
    gray = image.convert("L")
    np_img = np.array(gray)
    # 2) Progowanie Otsu -> binarizacja
    _, thresh = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 3) Deskew (prostowanie tekstu)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    (h, w) = thresh.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    # 4) Konwersja z powrotem do RGB
    return Image.fromarray(rotated).convert("RGB")

# Ścieżka do Twojego pliku (HEIC z rozszerzeniem .png)
image_path = Path(r"C:/Users/darek/Downloads/Zrzut ekranu 2025-06-24 144729.jpg")
if not image_path.exists():
    raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

# 1. HEIC → PIL.Image
heif_file = pillow_heif.read_heif(str(image_path))
img = Image.frombytes(
    heif_file.mode,
    heif_file.size,
    heif_file.data,
    "raw",
    heif_file.mode,
    heif_file.stride,
)

# 2. Pre-processing
img_pp = preprocess_pil(img)

# 3. Załaduj TrOCR-Large-Handwritten
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten",
    torch_dtype=torch.float16        # FP16 znacznie obniża zużycie VRAM
).to("cuda")                        # lub .to("cpu") jeśli chcesz CPU

# 4. Przygotowanie i inferencja
pixel_values = processor(images=img_pp, return_tensors="pt").pixel_values.to(model.device)
generated_ids = model.generate(
    pixel_values,
    max_length=512,
    num_beams=5,
    early_stopping=True
)

# 5. Dekodowanie i wyjście
recognized = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Rozpoznany tekst:\n", recognized.strip())



import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def preprocess_pil(image: Image.Image) -> Image.Image:
    # 1) Konwersja na skalę szarości
    gray = image.convert("L")
    np_img = np.array(gray)
    # 2) Progowanie Otsu → binarizacja
    _, thresh = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 3) Deskew (prostowanie tekstu)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    (h, w) = thresh.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(
        thresh, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    # 4) Z powrotem na RGB
    return Image.fromarray(rotated).convert("RGB")

# Ścieżka do Twojego pliku JPG
image_path = Path(r"C:\Users\darek\Downloads\Zrzut ekranu 2025-06-24 144729.jpg")
if not image_path.exists():
    raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

# 1. Wczytaj obraz jako JPG przez PIL
img = Image.open(image_path).convert("RGB")

# 2. Pre‐processing (grayscale + Otsu + deskew)
img_pp = preprocess_pil(img)

# 3. Załaduj TrOCR‐Large‐Handwritten
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten",
    torch_dtype=torch.float16
).to("cuda")  # lub "cpu" jeśli brak GPU

# 4. Przygotowanie i inferencja
pixel_values = processor(images=img_pp, return_tensors="pt").pixel_values.to(model.device)
generated_ids = model.generate(
    pixel_values,
    max_length=512,
    num_beams=5,
    early_stopping=True
)

# 5. Dekodowanie wyniku
recognized = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Rozpoznany tekst:\n", recognized.strip())

import cv2
import pytesseract
from pathlib import Path

# (opcjonalnie, ale już niekonieczne – masz Tesseract w PATH)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 1. Wczytaj obraz w odcieniach szarości
image_path = Path(r"C:\Users\darek\Downloads\Zrzut ekranu 2025-06-24 144729.jpg")
img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

# 2. Binarizacja Otsu (usuwa tło, poprawia kontrast)
_, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. (opcjonalnie) deskew – pomija jeśli tekst prosty
coords = cv2.findNonZero(img_bin)
if coords is not None:
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    (h, w) = img_bin.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img_bin = cv2.warpAffine(img_bin, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

# 4. OCR po polsku
text = pytesseract.image_to_string(img_bin, lang='pol', config='--psm 6')

print("Rozpoznany tekst:\n", text)

