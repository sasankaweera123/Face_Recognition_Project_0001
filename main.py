import cv2
import numpy as np
import face_recognition

# This will be the image going to train
imgSasa = face_recognition.load_image_file('IMG/Sasa_1.jpeg')
imgSasa = cv2.cvtColor(imgSasa, cv2.COLOR_BGR2RGB)

# This is the image going to test
imgTest = face_recognition.load_image_file('IMG/Sasa_3.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSasa)[0]
encodeSasa = face_recognition.face_encodings(imgSasa)[0]
cv2.rectangle(imgSasa, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,255,0),2)
# print(faceLoc)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeSasaTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,255,0),2)


results = face_recognition.compare_faces([encodeSasa],encodeSasaTest)
faceDis = face_recognition.face_distance([encodeSasa],encodeSasaTest)
print(results,faceDis)

status = 'Matched'
if results[0]:
    status = 'Matched'
else:
    status = 'UnMatched'

cv2.putText(imgTest,f'{status} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)



cv2.imshow('Sasa Image', imgSasa)
cv2.imshow('Sasa Image Test', imgTest)
cv2.waitKey(0)
