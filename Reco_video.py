import cv2
import operator
import pickle
from numpy import *
from imutils.object_detection import non_max_suppression
import time
global label, rect


def reco_visage_vd():
    index = 0
    marge = 70
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        width = frame.shape[1]
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
        flip_frame = cv2.flip(frame, 1)
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4)
        profile = profile_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4)
        profile_flip = profile_cascade.detectMultiScale(flip_frame, scaleFactor=1.4, minNeighbors=4)
        faces_2 = []

        for (x, y, w, h) in faces:
            faces_2.append([x, y, x + w, y + h])
        for (x, y, w, h) in profile_flip:
            faces_2.append([width - x, y, width - (x + w), y + h])
        for (x, y, w, h) in profile:
            faces_2.append([x, y, x + w, y + h])
        faces_2 = sorted(faces_2, key=operator.itemgetter(0, 1))
        for (x, y, w, h) in faces_2:
            if not index or (x - faces[index - 1][0] > marge or y - faces_2[index - 1][1] > marge):
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break


def reco_pers():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    reco = cv2.face.LBPHFaceRecognizer_create()
    reco.read("trainner.yml")
    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flip_frame = cv2.flip(gray, 1)
        faces = array(face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50)))
        profile = array(profile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50)))
        profile_flip = array(
            profile_cascade.detectMultiScale(flip_frame, scaleFactor=1.4, minNeighbors=4, minSize=(50, 50)))
        # (rects, weights) = array(hog.detectMultiScale(img_hsv, winStride=(4, 4), padding=(8, 8), scale=1.05))
        # print(faces, profile_flip)
        # tout = concatenate((faces, profile))
        # tout = array([[x, y, x + w, y + h] for (x, y, w, h) in tout])
        # tout = concatenate((tout, array([frame.shape[1] - x, y, frame.shape[1] - (x + w), y + h] for (x, y, w, h) in profile_flip)))
        # pick = non_max_suppression(tout, probs=None, overlapThresh=0.65)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = reco.predict(roi_gray)
            if conf <= 80:
                color = (255, 0, 0)
                name = labels[id_]
            else:
                color = (0, 255, 0)
                name = "Inconnu"
            label = name + " " + '{:5.2f}'.format(conf)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        for (x, y, w, h) in profile:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = reco.predict(roi_gray)
            if conf <= 80:
                color = (255, 0, 0)
                name = labels[id_]
            else:
                color = (0, 255, 0)
                name = "Inconnu"
            label = name + " " + '{:5.2f}'.format(conf)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        for (x, y, w, h) in profile_flip:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = reco.predict(roi_gray)
            if conf <= 80:
                color = (255, 0, 0)
                name = labels[id_]
            else:
                color = (0, 255, 0)
                name = "Inconnu"
            label = name + " " + '{:5.2f}'.format(conf)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (frame.shape[1] - x, y), (frame.shape[1] - (x + w), y + h), color, 2)

        # for (xA, yA, xB, yB) in rects:
        #     cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        #
        # cv2.imshow('vid', frame)
        #
        # if cv2.waitKey(1) == ord('q'):
        #    break


def reco_pers2():
    i = 0
    duree = []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    reco = cv2.face.LBPHFaceRecognizer_create()
    reco.read("trainner.yml")
    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    cap = cv2.VideoCapture(0)
    width = int(cap.get(3))
    while True:
        d = time.time()
        tickmark = cv2.getTickCount()
        if i % 5 == 0:
            label = []
            faces_all = []
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flip_frame = cv2.flip(gray, 1)
            faces = array(face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50)))
            profile = array(profile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50)))
            profile_flip = array(
                profile_cascade.detectMultiScale(flip_frame, scaleFactor=1.4, minNeighbors=4, minSize=(50, 50)))
            for (x, y, w, h) in faces:
                faces_all.append([x, y, x + w, y + h])
            for (x, y, w, h) in profile_flip:
                faces_all.append([width - (x + w), y, width - x, y + h])
            for (x, y, w, h) in profile:
                faces_all.append([x, y, x + w, y + h])
            faces_all = array(faces_all)
            rect = non_max_suppression(faces_all, probs=None, overlapThresh=0.65)
            for (x, y, w, h) in rect:
                roi_gray = gray[y: h, x: w]
                id_, conf = reco.predict(roi_gray)
                if conf <= 80:
                    name = labels[id_]
                else:
                    name = "Inconnu"
                label.append(name + " " + '{:5.2f}'.format(conf))
        f = time.time()
        duree.append(f - d)
        i += 1
        if i == 1000:
            break
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - tickmark)
        # if fps < 35:
        #         #     print("fps: {:05.2f}".format(fps))
    cv2.destroyAllWindows()
    return duree

    # on envoie juste le label et rect, il faut, au niveau de l'application dessiner sur l'image le rectangle
    # de coordonnées ((x début, y début), (x fin, h fin)) qui est pour un i dans range(len(rect))
    # ((rect[i][0], rect[i][1]), (rect[i][2], rect[i][3]))
    # puis il faut écrire le nom ou inconnu au dessus du rectangle à la position
    # (rect[i][0], rect[i][0]) et écrire label[i]


if __name__ == "__main__":
    duree = reco_pers2()




