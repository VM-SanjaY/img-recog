from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pytesseract

image_file = "D:\\python\\text recognition\\images\\1040.png"

# im = Image.open(image_file) # using pillow to read the file
# img = cv2.imread(image_file) # using open-cv to read the file
# cv2.imshow("Original Image",img)  # this will show image with title Original Image
# cv2.waitKey(0) # wait for indefinite time
# print(im)                 # this will provide pil info about the image
# print(im.size)            # this will provide the size of the image
# im.show()                 # This will show the image in new tab
# im.rotate(90).show()      # this will rotate the image to 90,180,270,360*
# im.save("D:\\python\\text recognition\\images\\new1040.png") # to save the image

# ---------------------------------------------------------------------------------------------------

# ------------------ This function shows the image in correct size...................................                

#https://stackoverflow.com/questions/28816046/displaying-different-images-with-actual-size-in-matplotlib-subplot
# def display(im_path):
#     dpi = 80  # dots-per-inch (pixel per inch).
#     im_data = plt.imread(im_path)

#     height, width  = im_data.shape[:2]
    
#     # What size does the figure need to be in inches to fit the image?
#     figsize = width / float(dpi), height / float(dpi)

#     # Create a figure of the right size with one axes that takes up the full figure
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_axes([0, 0, 1, 1])

#     # Hide spines, ticks, etc.
#     ax.axis('off')

#     # Display the image.
#     ax.imshow(im_data, cmap='gray')

#     plt.show()

# display(image_file)

# ------------------------------------------------------------------------------------------------------------------
    #  ********************************      Preprocessing The image    *************************************************
# ------------------------------------- Inverted Images -------------------------------------------------------

# it is best to avoid using this as the newer version do not need this

# Best suited for vesion 3 and below

# inverted_image = cv2.bitwise_not(img) # this method will invert the image
# cv2.imwrite("D:\\python\\text recognition\\images\\inverted.jpg", inverted_image) # this will save the image to the path

# display("D:\\python\\text recognition\\images\\inverted.jpg") # the inverted image will be shown using the above function


# ----------------------------------------------------------------------------------------------------------------------------------

# -------------------------------Binarization -----------------------------------------------------------------------------------------

# this is important for most of the py tesseract extraction code. Try playing with the number to get a
# clear image to extract (gray_image, 210, 230, cv2.THRESH_BINARY) = > numbers here

# def grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Tell they color you want to convert it to
                                                     # It reads the image in BGR format and not RGB
# gray_image = grayscale(img)
# cv2.imwrite("D:\\python\\text recognition\\images\\gray.jpg", gray_image)

# display("D:\\python\\text recognition\\images\\gray.jpg")

# thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY) # this will make the gray image bolder and it need two integers 
# cv2.imwrite("D:\\python\\text recognition\\images\\bw_image.jpg", im_bw) # there are more 5 methods to use. try change number above to see the difference


# ---------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------ Noise removal -------------------------------------------------------------------------------------------------

# noise removal is only used to remove unwanted dots. If image is good dont use it 

# def noise_removal(image):
#     kernel = np.ones((1, 1), np.uint8) # 1,1 are the shape
#     image = cv2.dilate(image, kernel, iterations=1) # it takes image and get size from kernal and loop it 1 time
#     kernel = np.ones((1, 1), np.uint8) # it need unother kernal for erodetion
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) # next we have to combine both the image buth make sure you name them the same
#     image = cv2.medianBlur(image, 3) # try change the number to see the blur quality
#     return (image)


# no_noise = noise_removal(im_bw) # the image out of thresh, im_bw is mostly prefred but we can also you normal image has well
# cv2.imwrite("D:\\python\\text recognition\\images\\no_noise.jpg", no_noise)
# display("D:\\python\\text recognition\\images\\no_noise.jpg")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------- Dilation and Erosion -----------------------------------------------------------------------------------------------------------------------------------

# Erosion - converting the image text to thin

# def thin_font(image):
#     image = cv2.bitwise_not(image) # this is important for erode and dilate to work
#     kernel = np.ones((2,2),np.uint8)
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.bitwise_not(image) and image should be converted again to get the same color image as before with the erode or dilate involved
#     return (image)

# eroded_image = thin_font(no_noise)  # we can use any image and not just no_noise image
# cv2.imwrite("D:\\python\\text recognition\\images\\eroded_image.jpg", eroded_image)

# display("D:\\python\\text recognition\\images\\eroded_image.jpg")


# # Dilation - converting the image text to thick

# def thick_font(image):
#     image = cv2.bitwise_not(image)
#     kernel = np.ones((2,2),np.uint8)
#     image = cv2.dilate(image, kernel, iterations=1)
#     image = cv2.bitwise_not(image)
#     return (image)


# dilated_image = thick_font(no_noise)
# cv2.imwrite("D:\\python\\text recognition\\images\\dilated_image.jpg", dilated_image)

# display("D:\\python\\text recognition\\images\\dilated_image.jpg")


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------Rotation / Deskewing ------------------------------------------------------


# This function will set the image striaght if it is slanted to the right or left. based on bounding boxs
# for this function to work make sure there are no borders. if there are remove it before using it

# #https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df

# def getSkewAngle(cvImage) -> float:
#     # Prep image, copy, convert to gray scale, blur, and threshold
#     newImage = cvImage.copy()
#     gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (9, 9), 0)
#     thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#     # Apply dilate to merge text into meaningful lines/paragraphs.
#     # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
#     # But use smaller kernel on Y axis to separate between different blocks of text
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
#     dilate = cv2.dilate(thresh, kernel, iterations=2)

#     # Find all contours
#     contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key = cv2.contourArea, reverse = True)
#     for c in contours:
#         rect = cv2.boundingRect(c)
#         x,y,w,h = rect
#         cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

#     # Find largest contour and surround in min area box
#     largestContour = contours[0]
#     print (len(contours))
#     minAreaRect = cv2.minAreaRect(largestContour)
#     cv2.imwrite("temp/boxes.jpg", newImage)
#     # Determine the angle. Convert it to the value that was originally used to obtain skewed image
#     angle = minAreaRect[-1]
#     if angle < -45:
#         angle = 90 + angle
#     return -1.0 * angle
# # Rotate the image around its center
# def rotateImage(cvImage, angle: float):
#     newImage = cvImage.copy()
#     (h, w) = newImage.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return newImage


# def deskew(cvImage):
#     angle = getSkewAngle(cvImage)
#     return rotateImage(cvImage, -1.0 * angle)

# fixed = deskew(new) # new here is the slanted image but i do not have a complet of it
# cv2.imwrite("D:\\python\\text recognition\\images\\rotated_fixed.jpg", fixed) # it fixes the image to straight

# display("D:\\python\\text recognition\\images\\rotated_fixed.jpg")

# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# this function will remove the borders if the borders are available

# def remove_borders(image):
#     contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
#     cnt = cntsSorted[-1]
#     x, y, w, h = cv2.boundingRect(cnt)
#     crop = image[y:y+h, x:x+w]
#     return (crop)


# no_borders = remove_borders(no_noise)
# cv2.imwrite("D:\\python\\text recognition\\images\\no_borders.jpg", no_borders)
# display('D:\\python\\text recognition\\images\\no_borders.jpg')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------- Adding Border ----------------------------------------------------------------------------------------------------------------------

# the below function will create a white border with 150px at all sides

# color = [255, 255, 255] # border color
# top, bottom, left, right = [150]*4  #  border size

# image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
# cv2.imwrite("D:\\python\\text recognition\\images\\image_with_border.jpg", image_with_border)
# display("D:\\python\\text recognition\\images\\image_with_border.jpg")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# topics missed =  Transparency / Alpha Channel and Rescaling


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# using Py tesseract to extract text from image

img = Image.open(image_file)

ocr_result = pytesseract.image_to_string(img,lang='eng')

print(ocr_result)


















