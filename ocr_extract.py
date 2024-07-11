import cv2
import pytesseract
from PIL import Image
import os
import spacy
from nltk.tokenize import word_tokenize

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize spacy NLP model
nlp = spacy.load('en_core_web_sm')

# Function to perform OCR
def perform_ocr(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use pytesseract to perform OCR on the image
    text = pytesseract.image_to_string(gray)
    return text

# Function to clean extracted text
def clean_text(text):
    # Tokenize text using nltk
    tokens = word_tokenize(text)
    
    # Remove non-alphabetic tokens
    words = [word for word in tokens if word.isalpha()]
    
    # Join words back into a single string
    cleaned_text = ' '.join(words)
    return cleaned_text

# Function to annotate text
def annotate_text(text):
    doc = nlp(text)
    annotations = {
        'named_entities': [(ent.text, ent.label_) for ent in doc.ents],
        'pos_tags': [(token.text, token.pos_) for token in doc]
    }
    return annotations

# Path to the main directory containing subfolders
main_dir = "dataset"

# Process each subfolder in the main directory
for category in os.listdir(main_dir):
    category_path = os.path.join(main_dir, category)
    
    # Ensure the current path is a directory
    if os.path.isdir(category_path):
        print(f"Processing category: {category}")
        
        # Process each image in the subfolder
        for image_file in os.listdir(category_path):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(category_path, image_file)
                
                # Perform OCR
                text = perform_ocr(image_path)
                
                # Clean the extracted text
                cleaned_text = clean_text(text)
                
                # Annotate the text
                annotations = annotate_text(cleaned_text)
                
                # Print results
                print(f"Image: {image_file}")
                print(f"Cleaned Text: {cleaned_text}")
                print(f"Annotations: {annotations}")
                print("\n")

# Save this script as ocr_processing.py in the directory C:\Users\91767\Documents\GitHub\OCR-processing
