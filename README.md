# 📄 Urdu-OCR System

A Python-based OCR system for extracting **Urdu text** from PDFs and images using Tesseract, with preprocessing optimizations for Naskh Urdu scripts.

---

## 🚀 Approach & Design Decisions
- **Approach**:  
  Developed a **Python-based OCR system** leveraging **Tesseract** to extract Urdu text from PDFs and images.
- **Key Design Decisions**:  
  - Used **Google Colab** for execution due to free GPU/CPU resources and pre-installed libraries.
  - Applied **adaptive thresholding** for binarization to handle Naskh Urdu variations.
  - Skipped memory-heavy deskewing after repeated Colab crashes to maintain stability.

---

## 🛠️ Tools, Frameworks & Libraries

| Tool / Library | Purpose |
|----------------|---------|
| **Tesseract-OCR** | Core OCR engine with Urdu language support |
| **OpenCV** | Image preprocessing (noise removal, CLAHE, thresholding) |
| **Pillow (PIL)** | Image handling |
| **pdf2image** | Convert PDFs to images |
| **jiwer** | WER/CER evaluation |
| **Google Colab** | Development environment |

---

## 📂 Dataset(s) Used / Created
- **Primary Dataset**: `Urdu.pdf` (Naskh Urdu text)
- **Evaluation File**: `ground_truth.txt` for accuracy testing

---

## 🖼️ Preprocessing Pipeline
1. **Grayscale Conversion** – Simplifies image data  
2. **Noise Removal** – Using `fastNlMeansDenoising`  
3. **Contrast Enhancement** – Via **CLAHE**  
4. **Adaptive Thresholding** – For binarization and text clarity  

---

## 📊 Accuracy Results & Outputs
- **Initial Accuracy**:  
  - **WER**: 24%  
  - **CER**: ~11%  
- **Optimization Target**: WER < 20%  
- **Sample Output**:  
  - Extracted text saved in: `extracted_urdu_text.txt`  
  - Compared against: `ground_truth.txt`

---

## ⚠️ Challenges Faced
- Frequent **Google Colab crashes** due to memory limits  
- **Inconsistent Naskh Urdu recognition** in certain fonts  
- Lack of **font-specific training data**

---

## 🔮 Future Improvements
- Train **Tesseract** with a **Naskh-specific dataset**
- Add **deskewing** with memory-optimized processing
- Move to a **more powerful cloud service** (AWS, GCP, Azure)


