# Środowisko Python – LoRA 4bit (Windows + GPU)

To repozytorium zawiera snapshot mojego działającego środowiska Python pod Windows,
na którym poprawnie działają modele z obsługą LoRA 4bit, `bitsandbytes`, `transformers` itp.

Celem jest możliwość odtworzenia środowiska lub porównania go z innymi konfiguracjami.

---

## Wersje kluczowych komponentów

### Python

- Wersja: **Python 3.11.1**

```powershell
python --version
# Python 3.11.1
```

### Pip

- Wersja: **pip 25.1.1**
- Ścieżka instalacji:

```text
C:\Users\darek\AppData\Local\Programs\Python\Python311\Lib\site-packages\pip
```

```powershell
python -m pip --version
# pip 25.1.1 from C:\Users\darek\AppData\Local\Programs\Python\Python311\Lib\site-packages\pip (python 3.11)
```

---

## GPU / NVIDIA

- GPU: **NVIDIA GeForce RTX 4070**
- Sterowniki: **Driver Version 581.80**
- CUDA (wg sterownika): **CUDA Version 13.0**
- Pamięć GPU: **16 GB (16376 MiB)**

Fragment wyjścia `nvidia-smi`:

```text
NVIDIA-SMI 581.80                 Driver Version: 581.80         CUDA Version: 13.0

GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC
  0  NVIDIA GeForce RTX 4070 ...  WDDM  |   00000000:26:00.0  On |                  N/A
     Fan  Temp   Perf          Pwr:Usage/Cap
     0%   40C    P8             14W /  285W

     Memory-Usage: 2084MiB / 16376MiB
     GPU-Util: 0%
```

---

## Pliki w repozytorium

- `requirements_lora4bit.txt`  
  Snapshot wszystkich zainstalowanych pakietów Pythona i ich wersji, wygenerowany poleceniem:

  ```powershell
  python -m pip freeze > requirements_lora4bit.txt
  ```

- `env_info_lora4bit.txt`  
  Dodatkowe informacje o środowisku:
  - wersja Pythona
  - wersja pip
  - wyjście `nvidia-smi`
  - notatki o tym, że środowisko NIE jest venv (globalny Python 3.11 na Windows)

---

## Jak odtworzyć środowisko (z grubsza)

> Najlepiej robić to w **nowym, odseparowanym wirtualnym środowisku** (venv).

Przykład (Windows, PowerShell):

```powershell
# 1. Utwórz i aktytywuj wirtualne środowisko
python -m venv venv
.env\Scriptsctivate

# 2. Zainstaluj pakiety z pliku
python -m pip install --upgrade pip
python -m pip install -r requirements_lora4bit.txt
```

W zależności od sprzętu / sterowników GPU niektóre paczki (np. `torch`, `bitsandbytes`, `onnxruntime`, `openvino`) mogą wymagać dostosowania do lokalnej konfiguracji.

---

## Uwagi

- Środowisko powstało na **Windows** z globalnym Pythonem 3.11.1.
- Repo to pełni rolę snapshotu referencyjnego do debugowania lub odtwarzania konfiguracji.
