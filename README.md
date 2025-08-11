## Documentation

### Approach and Design Decisions
- **Approach**: Developed a Python-based OCR system using Tesseract for Urdu text extraction from PDFs and images.
- **Design Decisions**: Used Google Colab for execution due to its free resources and pre-installed libraries. Chose adaptive thresholding for binarization to handle Naskh Urdu variations, avoiding memory-intensive deskewing after crashes.

### Tools, Frameworks, and Libraries
- **Tesseract-OCR**: Core OCR engine with Urdu support.
- **OpenCV**: For image preprocessing (noise removal, CLAHE, thresholding).
- **Pillow**: For image handling.
- **pdf2image**: To convert PDFs to images.
- **jiwer**: For WER/CER evaluation.
- **Flask**: For web deployment on Vercel.
- **Google Colab**: Development environment.

### Dataset(s) Used or Created
- **Dataset**: Single `Urdu.pdf` file with Naskh Urdu text, supplemented by a `ground_truth.txt` for evaluation.

### Preprocessing Steps
- Convert image to grayscale.
- Apply `fastNlMeansDenoising` for noise removal (h=5, templateWindowSize=5, searchWindowSize=15).
- Use CLAHE (clipLimit=5.0, tileGridSize=(8, 8)) for contrast enhancement.
- Apply adaptive thresholding for binarization.

### Accuracy Results and Sample Outputs
- **Results**: Achieved WER of 24% and CER of ~11% (initial); post-optimization target is WER < 20%.
- **Sample Output**: `extracted_urdu_text.txt` contains extracted text; compare with `ground_truth.txt`.

### Challenges Faced and Future Improvements
- **Challenges**: Google Colab crashes due to memory limits, inconsistent Naskh recognition, and lack of font-specific training.
- **Future Improvements**: Train Tesseract with a Naskh dataset, add deskewing with memory optimization, or use a more powerful cloud service (e.g., AWS).
