import cv2

# Read the input image
img = cv2.imread('data/face1.jpg')
# print(img)
# cv2.imshow('face', img)
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
print(face_cascade)
# Detect faces
faces = face_cascade.detectMultiScale(gray)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h),
                  (0, 0, 255), 2)

    faces = img[y:y + h, x:x + w]
    cv2.imshow("face", faces)
    cv2.imwrite('face.jpg', faces)

cv2.imshow('img', img)
cv2.waitKey()