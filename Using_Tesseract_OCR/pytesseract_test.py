from PIL import Image
import pytesseract
import cv2
import os


img_path = "images/example_thresh.png"
preprocess = "thresh"

# img_path = "images/receipt.png"
# preprocess = "blur"

image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply thresholding to segment the foreground from the background
# 这种阈值方法对于读取覆盖在灰色形状上的暗文本非常有用
if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# median blurring to remove salt and pepper noise
elif preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)

# save the grayscale image as a temporary file so we can apply OCR to it
filename = os.path.join("images", "postprocess.png")
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, then delete the temporary file
# convert the contents of the image into string
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
