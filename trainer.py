import os
import cv2
import numpy as np
import pickle


def trainer():
    cId = 0
    labelId = {}
    c = 50
    train = []
    ylab = []
    for i in os.walk("Elie"):
        if len(i[2]):
            label = i[0].split("/")[-1]
            for f in i[2]:
                if f.endswith("png"):
                    path = os.path.join(i[0], f)
                    if label not in labelId:
                        labelId[label] = cId
                    id_ = labelId[label]
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    train.append(image)
                    ylab.append(id_)

    with open("labels.pickle", "wb") as f:
        pickle.dump(labelId, f)

    train = np.array(train)
    ylab = np.array(ylab)
    reco = cv2.face.LBPHFaceRecognizer_create()
    reco.train(train, ylab)
    reco.save("trainner.yml")


if __name__ == "__main__":
    trainer()
