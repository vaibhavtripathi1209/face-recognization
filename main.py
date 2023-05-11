# import python Libraries
import cv2
import face_recognition

# Load first Image in the system
img1 = face_recognition.load_image_file('group.jpeg')
# make Photosterio image
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Load second Image in the system
imgTest = face_recognition.load_image_file('sumit.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# make the rectangular box on the first face
faceLoc = face_recognition.face_locations(img1)[0]
encodeElon = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# make the rectangular box on the first face
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# make the rectangular box on the second face
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results)
cv2.putText(imgTest, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Show the Image on the screen
cv2.imshow('Elon Musk', img1)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
