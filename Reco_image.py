import cv2
from imutils.object_detection import non_max_suppression
import imutils
import pickle
import numpy as np


def reco_face_im(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    img = cv2.imread(image)
    flip_img = cv2.flip(img, 1)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=1)
    profile = profile_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2)
    profile_flip = profile_cascade.detectMultiScale(flip_img, scaleFactor=1.1, minNeighbors=2)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in profile_flip:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in profile:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_face_im(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    img = cv2.imread(image)
    flip_img = cv2.flip(img, 1)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=1)
    profile = profile_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2)
    profile_flip = profile_cascade.detectMultiScale(flip_img, scaleFactor=1.4, minNeighbors=2)
    extraction = [i * [] for i in range(len(faces) + len(profile) + len(profile_flip))]
    x, y, w, h = [], [], [], []
    for i in faces:
        x.append(i[1])
        y.append(i[0])
        w.append(i[2])
        h.append(i[3])
        print('face')
    for i in profile:
        x.append(i[1])
        y.append(i[0])
        w.append(i[2])
        h.append(i[3])
        print('profile')
    for i in profile_flip:
        x.append(i[1])
        y.append(i[0])
        w.append(i[2])
        h.append(i[3])
        print('profile_flip')
    for i in range(len(x)):
        extraction[i] = img[x[i]:x[i] + w[i], y[i]:y[i] + h[i]]

    for i in range(len(extraction)):
        cv2.imwrite('visage_n%s.png' % i, extraction[i])


def reco_fullbody_img_haar(image):
    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    img = cv2.imread(image)
    img = cv2.resize(img, (640, 360))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fullbody = body_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in fullbody])
    pick = non_max_suppression(rects)
    for (x, y, w, h) in pick:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def reco_fullbody_img_hog(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    img = cv2.imread(image)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fullbody = hog.detectMultiScale(img_hsv, winStride=(4, 4), padding=(8, 8), scale=1.01)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in fullbody[0]])
    pick = non_max_suppression(rects)
    for (x, y, w, h) in pick:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def reco_face_fullbody_img(image):
    reco = cv2.face.LBPHFaceRecognizer_create()
    reco.read("trainner.yml")
    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    img = cv2.imread(image)
    img = imutils.resize(img, width=min(400, img.shape[1]))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (rects, weights) = hog.detectMultiScale(img_hsv, winStride=(4, 4), padding=(8, 8), scale=1.01)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.05)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = img_gray[y:y + h, x:x + w]
        id_, conf = reco.predict(roi_gray)
        if conf <= 80:
            color = (0, 255, 0)
            name = labels[id_]
        else:
            color = (0, 0, 255)
            name = "Inconnu"
        label = name + " " + '{:5.2f}'.format(conf)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = 'moi.jpg'
    reco_face_fullbody_img(image)