import cv2
import numpy as np
import face_recognition
import os


def get_classnames():
    for cl in myList:
        current_image = cv2.imread(f'{path}/{cl}')
        images.append(current_image)
        classNames.append(os.path.splitext(cl)[0])
    return images, classNames


def find_name(index, image):
    if matches[index]:
        name = classNames[index].upper()
        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, y2 - 35), (x2, y2), cv2.FILLED)
        cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        return image


# This will be the image going to train
def find_encodings(image):
    encode_list = []
    for i in image:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(i)[0]
        encode_list.append(encode)
    return encode_list


def get_c_ec_frame(capital):
    success, img = capital.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

    return img,imgS, faceCurrentFrame, encodeCurrentFrame


# main program starts here
if __name__ == "__main__":
    path = 'IMG/Attendence'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    images, classNames = get_classnames()
    print(classNames)
    encodeListKnown = find_encodings(images)
    # print(len(encodeListKnown))
    cap = cv2.VideoCapture(0)

    while True:

        img, imgS, faceCurrentFrame, encodeCurrentFrame = get_c_ec_frame(cap)

        for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            img = find_name(matchIndex, img)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)


