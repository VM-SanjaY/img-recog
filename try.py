import cv2
import numpy as np
import pytesseract


size = (1080, 640)
img = cv2.imread('unblurred.png')


img = cv2.resize(img, size, fx=0.5, fy=0.5)
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray, lang='eng')

sensitive_data_coords1 = (855, 250, 200, 35)
sensitive_data_coords2 = (855, 312, 200, 35)

x, y, w, h = sensitive_data_coords1
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)
cropped_img = img[y:y+h, x:x+w]
if cropped_img.size == 0:
    print("Error: blur cord is outside the image")
else:
    blur = cv2.blur(cropped_img, (50, 50))
    img[y:y+h, x:x+w] = blur

x, y, w, h = sensitive_data_coords2
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)
cropped_img = img[y:y+h, x:x+w]
if cropped_img.size == 0:
    print("Error: blur cord is outside the image")
else:
    blur = cv2.blur(cropped_img, (50, 50))
    img[y:y+h, x:x+w] = blur


cv2.imwrite('temp/syed.jpg', img)