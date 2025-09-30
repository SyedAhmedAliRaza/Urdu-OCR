!pip install pytesseract
!pip install opencv-python
!pip install pillow
!pip install pdf2image
!sudo apt-get install tesseract-ocr tesseract-ocr-urd poppler-utils -y
!pip install jiwer

from google.colab import files
import cv2
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import numpy as np

def preprocess_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=5, searchWindowSize=15)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(binary)

def load_and_preprocess(input_file):
    try:
        if input_file.lower().endswith('.pdf'):
            images = convert_from_path(input_file, dpi=200)
        else:
            images = [Image.open(input_file)]
        preprocessed_images = [preprocess_image(img) for img in images]
        print(f"Loaded and preprocessed {len(preprocessed_images)} images.")
        return preprocessed_images
    except Exception as e:
        print(f"Error loading or preprocessing: {e}")
        return []

uploaded = files.upload()

input_file = 'Urdu.pdf'
preprocessed_images = load_and_preprocess(input_file)
if not preprocessed_images:
    print("No images processed. Check file name or content.")
    exit()

def extract_urdu_text(images):
    full_text = ""
    for i, img in enumerate(images):
        print(f"Processing image {i+1} with shape: {np.array(img).shape}...")
        text = pytesseract.image_to_string(img, lang='urd', config='--oem 1 --psm 3')
        if text.strip():
            full_text += f"--- Page/Image {i+1} ---\n{text}\n"
            print(f"Detected text: {text.strip()}")
        else:
            full_text += f"--- Page/Image {i+1} ---\nNo text detected.\n"
    return full_text

extracted_text = extract_urdu_text(preprocessed_images)
print(extracted_text)

with open('extracted_urdu_text.txt', 'w', encoding='utf-8') as f:
    f.write(extracted_text)

files.download('extracted_urdu_text.txt')

try:
    from jiwer import wer, cer
    with open('extracted_urdu_text.txt', 'r', encoding='utf-8') as f:
        extracted = f.read().strip()
    ground_truth_file = 'ground_truth.txt'
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        truth = f.read().strip()
    word_error_rate = wer(truth, extracted)
    char_error_rate = cer(truth, extracted)
    print(f"Word Error Rate (WER): {word_error_rate:.2%}")
    print(f"Character Error Rate (CER): {char_error_rate:.2%}")
except FileNotFoundError:
    print("Ground truth file not found. Please upload ground_truth.txt.")
except Exception as e:
    print(f"Error during evaluation: {e}")
