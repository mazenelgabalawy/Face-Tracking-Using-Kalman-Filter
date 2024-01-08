import cv2
import serial
import time
from KalmanFilter import KalmanFilter


def main():

    # Create opencv video capture object
    VideoCap = cv2.VideoCapture(1)
    #Create Arduino Serial Object
    ArduinoData = serial.Serial(port='COM3', baudrate=115200)

    #Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)

    #Create Classifier object
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    

    while(True):
        # Read frame
        ret, frame = VideoCap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width  = VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 191, 255), 2)
            centerx, centery = ((x+x+w)/2, (y+y+h)/2)
            rectcenter = int(centerx),int(centery)
            cv2.circle(frame, rectcenter, 5 , (0, 191, 255), 3)
        
        # If centroids are detected then track them
        if (len(faces) > 0):

            # Predict
            (x_p, y_p) = KF.predict()
            # Draw a rectangle as the predicted object position
            cv2.rectangle(frame, (int(x_p - w/2), int(y_p - h/2)), (int(x_p + w/2), int(y_p + h/2)), (0,255, 0), 2)

            # Update
            (x1, y1) = KF.update([[centerx],[centery]])

            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (int(x1 - w/2), int(y1 - h/2)), (int(x1 + w/2), int(y1 + h/2)), (0, 0, 255), 2)
            cv2.circle(frame, (int(x1),int(y1)) , 5,(0,0,255),3)


            cv2.putText(frame, "Estimated Position", (int(x1 + 30), int(y1 + 30)), 0, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (int(x_p + 15), int(y_p)), 0, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Measured Position", (int(centerx + 15), int(centery - 15)), 0, 0.8, (0,191,255), 2)


            #Send Center Data to Arduino
            ArduinoSentData=str(int(width))+','+ str(int(x1))+'\r'
            print(ArduinoSentData)
            ArduinoData.write(ArduinoSentData.encode())
            time.sleep(0.1)

        cv2.imshow('image', frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    # execute main
    main()
