

# Tesseract OCR library
!pip install pytesseract
# OpenCV for image processing
!pip install opencv-python
# Pillow for image handling
!pip install pillow
# pdf2image to convert PDF to images
!pip install pdf2image
# Tesseract and Urdu support, plus PDF utilities
!sudo apt-get install tesseract-ocr tesseract-ocr-urd poppler-utils -y
# jiwer for calculating word and character error rates
!pip install jiwer


# Importing files module from Google Colab
from google.colab import files
# Importing OpenCV library
import cv2
# Importing pdf2image
from pdf2image import convert_from_path
# Importing PIL
from PIL import Image
# Importing pytesseract
import pytesseract
# Importing numpy
import numpy as np


# Function to preprocess the image
def preprocess_image(image):
    # Converting the input image to a numpy array
    img_array = np.array(image)
    # Checking if the image is already grayscale or has one channel
    if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
        # Using the image as grayscale if condition is true
        gray = img_array
    else:
        # Converting the image to grayscale if it has multiple channels
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Removeing noise from the grayscale image with lighter settings
    gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=5, searchWindowSize=15)
    # Creating a CLAHE object to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    # Applying contrast enhancement to the grayscale image
    enhanced = clahe.apply(gray)
    # Converting the image to black and white using adaptive thresholding
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    # Returing the processed image as a PIL image
    return Image.fromarray(binary)


# Function to load and preprocess the input file
def load_and_preprocess(input_file):
    try:
        # CheckING if the input file is a PDF
        if input_file.lower().endswith('.pdf'):
            # Converting PDF to a list of images with 200 DPI
            images = convert_from_path(input_file, dpi=200)
        else:
            # Loading the file as a single image if not a PDF
            images = [Image.open(input_file)]
        # Processing each image using the preprocess_image function
        preprocessed_images = [preprocess_image(img) for img in images]
        # Printing the number of images processed
        print(f"Loaded and preprocessed {len(preprocessed_images)} images.")
        # Returning the list of preprocessed images
        return preprocessed_images
    except Exception as e:
        # Printing any error that occurs during loading or preprocessing
        print(f"Error loading or preprocessing: {e}")
        # Returning an empty list if an error occurs
        return []


# Uploading files to Colab
uploaded = files.upload()


# Setting the input file name to 'Urdu.pdf'
input_file = 'Urdu.pdf'
# Processing the input file and get preprocessed images
preprocessed_images = load_and_preprocess(input_file)
# Checking if any images were processed
if not preprocessed_images:
    print("No images processed. Check file name or content.")
    exit()


# Function to extract Urdu text from images
def extract_urdu_text(images):
    # Initializing an empty string to store the extracted text
    full_text = ""
    # For loop on each image
    for i, img in enumerate(images):
        # Printing the shape of the current image being processed
        print(f"Processing image {i+1} with shape: {np.array(img).shape}...")
        # Extracting text from the image using Tesseract with specific settings
        text = pytesseract.image_to_string(img, lang='urd', config='--oem 1 --psm 3')
        # Checking if any text was extracted
        if text.strip():
            # Adding the page number and extracted text to the result
            full_text += f"--- Page/Image {i+1} ---\n{text}\n"
            # Printing the detected text
            print(f"Detected text: {text.strip()}")
        else:
            # Adding a message if no text is detected
            full_text += f"--- Page/Image {i+1} ---\nNo text detected.\n"
    # Returning the complete text
    return full_text


# Extracting text from all preprocessed images
extracted_text = extract_urdu_text(preprocessed_images)
# Printing the extracted text
print(extracted_text)


# Opening a file
with open('extracted_urdu_text.txt', 'w', encoding='utf-8') as f:
    # Writing the extracted text to the file
    f.write(extracted_text)
# Downloading the text file
files.download('extracted_urdu_text.txt')


try:
    # Importing word error rate (wer) and character error rate (cer) functions
    from jiwer import wer, cer
    # Opening the extracted text file for reading
    with open('extracted_urdu_text.txt', 'r', encoding='utf-8') as f:
        # Reading and cleaning the extracted text
        extracted = f.read().strip()
    # Setting the ground truth file name
    ground_truth_file = 'ground_truth.txt'
    # Opening the ground truth file for reading
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        # Reading and cleaning the ground truth text
        truth = f.read().strip()
    # Calculating the word error rate
    word_error_rate = wer(truth, extracted)
    # Calculating the character error rate
    char_error_rate = cer(truth, extracted)
    # Printing the accuracy results
    print(f"Word Error Rate (WER): {word_error_rate:.2%}")
    print(f"Character Error Rate (CER): {char_error_rate:.2%}")
except FileNotFoundError:
    # Printing a message if the ground truth file is missing
    print("Ground truth file not found. Please upload ground_truth.txt.")
except Exception as e:
    # Printing any other errors during evaluation
    print(f"Error during evaluation: {e}")