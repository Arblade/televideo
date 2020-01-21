import cv2

global c
c = 50


def BDD():
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
    profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    cap = cv2.VideoCapture(0)
    id = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flip_frame = cv2.flip(gray, 1)
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(c, c))
        profile = profile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(c, c))
        flip_face = profile_cascade.detectMultiScale(flip_frame, scaleFactor=1.4, minNeighbors=4, minSize=(c, c))
        for x, y, w, h in face:
            cv2.imwrite("Elie/p-{:d}.png".format(id), frame[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            id += 1
        for x, y, w, h in profile:
            cv2.imwrite("Elie/p-{:d}.png".format(id), frame[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            id += 1
        for x, y, w, h in flip_face:
            cv2.imwrite("Elie/p-{:d}.png".format(id), frame[y:y + h, frame.shape[1] - x: frame.shape[1] - (x + w)])
            cv2.rectangle(frame, (frame.shape[1] - x, y), (frame.shape[1] - (x + w), y + h), (0, 0, 255), 2)
            id += 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        cv2.imshow('video', frame)


if __name__ == "__main__":
    BDD()
