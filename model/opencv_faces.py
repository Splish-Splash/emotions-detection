import cv2


def get_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    coords = face_cascade.detectMultiScale(gray)
    for i in range(len(coords)):
        coords[i] = (max(0, coords[i][0] - 20), max(0, coords[i][1] - 20), coords[i][2] + 20, coords[i][3] + 20)
    faces = []
    for i, (x, y, w, h) in enumerate(coords):
        face = img[y: y + h, x: x + w]
        faces.append(face)
    return faces, coords
