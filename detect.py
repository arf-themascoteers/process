from rembg.bg import remove
import numpy as np
import io
from PIL import Image, ImageFile
import cv2

input_path = 'raw/anick.jpg'
output_path = 'temp/anick.png'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread(input_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
it = iter(faces)
x, y, w, h = next(it)
center = (x + int(w/2), y + int(h/2))
modified_width_ratio = 1.5
modified_width = int(w * modified_width_ratio)
width_height_ratio = 1.2
modified_height = int(modified_width * width_height_ratio)

modified_top_left = (center[0] - int(modified_height/2), center[1] - int(modified_width/2))
modified_bottom_right = (modified_top_left[0] + modified_height, modified_top_left[1] + modified_width)

cropped_image = img[modified_top_left[0]:modified_bottom_right[0], modified_top_left[1]:modified_bottom_right[1]]
cropped_center = (int(cropped_image.shape[1]/2), int(cropped_image.shape[0]/2))
#cv2.rectangle(img,(x,y),(x+w,y+h), color=(255,0,255))
#cv2.rectangle(img,modified_top_left,modified_bottom_right, color=(255,255,0))

#cv2.circle(img, (eye_left_x,eye_left_y), radius=10, color=(255, 0, 255), thickness=5)
#cv2.circle(img, (eye_right_x,eye_right_y), radius=4, color=(0, 255, 0), thickness=1)
print(f"Width, Height: {cropped_image.shape}")

cv2.circle(cropped_image, cropped_center, radius=2, color=(255, 0, 255), thickness=5)

#cv2.imwrite("temp/anick.png",cropped_image)
cv2.imshow('img', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()