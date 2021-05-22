import cv2

img = cv2.imread("IMG_0006.jpg")
print(img.shape[0],img.shape[1])

img = cv2.resize(img,(200,100),interpolation=cv2.INTER_AREA)
cv2.imshow("img",img)
cv2.waitKey(0)