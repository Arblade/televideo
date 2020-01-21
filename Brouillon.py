import cv2

# person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# cap = cv2.VideoCapture('video.avi')
# while True:
#     r, frame = cap.read()
#     if r:
#         frame = cv2.resize(frame, (640, 360))  # Downscale to improve frame rate
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Haar-cascade classifier needs a grayscale image
#         rects = person_cascade.detectMultiScale(gray_frame, 1.05, 1)
#         for (x, y, w, h) in rects:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.imshow("preview", frame)
#     k = cv2.waitKey(1)
#     if k & 0xFF == ord("q"):  # Exit condition
#         break

# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# cap = cv2.VideoCapture('video.avi')
# while True:
#     r, frame = cap.read()
#     if r:
#         frame = cv2.resize(frame, (1280, 720))  # Downscale to improve frame rate
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image
#
#         rects, weights = hog.detectMultiScale(gray_frame)
#
#         for i, (x, y, w, h) in enumerate(rects):
#             if weights[i] < 0.7:
#                 continue
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         cv2.imshow("preview", frame)
#     k = cv2.waitKey(1)
#     if k & 0xFF == ord("q"):  # Exit condition
#         break


ret, frame = cap.read()
# img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
flip_frame = cv2.flip(gray, 1)
faces = array(face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50)))
profile = array(profile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50)))
profile_flip = array(profile_cascade.detectMultiScale(flip_frame, scaleFactor=1.4, minNeighbors=4, minSize=(50, 50)))
# (rects, weights) = array(hog.detectMultiScale(img_hsv, winStride=(4, 4), padding=(8, 8), scale=1.05))
if faces.shape[0] != 0 and profile.shape[0] != 0:
    tout = concatenate((faces, profile))
elif faces.shape[0] != 0 and profile.shape[0] == 0:
    tout = faces
else:
    tout = profile_flip
tout = array([[x, y, x + w, y + h] for (x, y, w, h) in tout])
print(tout)
# if profile_flip.shape[0] != 0:
#     tout = concatenate((tout, array([frame.shape[1] - x, y, frame.shape[1] - (x + w), y + h] for (x, y, w, h) in profile_flip)))
pick = non_max_suppression(tout, probs=None, overlapThresh=0.65)
print(array([frame.shape[1] - x, y, frame.shape[1] - (x + w), y + h] for (x, y, w, h) in profile_flip))
for (x, y, w, h) in pick:
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
