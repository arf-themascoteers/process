from rembg.bg import remove
import numpy as np
import io
from PIL import Image, ImageFile
import cv2

input_path = 'raw/anick.jpg'
output_path = 'temp/anick.png'

# Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
ImageFile.LOAD_TRUNCATED_IMAGES = True

f = np.fromfile(input_path)
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")

img.save(output_path)
img = cv2.imread(output_path)
# Load the cascade
face_cascade = cv2.CascadeClassifier('face_detector.xml')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect the faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw the rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# Display
cv2.imshow('img', img)
# Stop if escape key is pressed