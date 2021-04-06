# import the necessary packages
import numpy as np
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)
webcam = cv2.VideoCapture(1)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'Human_Detection.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret1, frame1 = webcam.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640,480))
    frame1 = cv2.resize(frame1, (640,480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    boxes1, weights = hog.detectMultiScale(frame1, winStride=(8,8) )

    boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
    boxes1 = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes1])

    person = 1
    person_webcam = 1

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (xA,yA), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
    for (xA, yA, xB, yB) in boxes1:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame1, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)
        cv2.putText(frame1, f'person {person_webcam}', (xA,yA), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person_webcam += 1

    # Write the output video
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('c922',frame1)

    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)

    cv2.putText(frame1, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame1, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
webcam.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)