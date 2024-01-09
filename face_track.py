import cv2
import serial
import time
from Kalman_Filter import *

# Define sampling time
dt = 0.1
# Define acceleration in x and y direction
u_x = 0.5
u_y = 0.5
# Define process noise
std_acc = 0.8
# Define standard deviation in measurement in x and y
x_std_meas = 0.1
y_std_meas = 0.1

# Define the  control input variables
U = np.matrix([[u_x],[u_y]])

# Intial State
X = np.matrix([[0], [0], [0], [0]])

# Define the State Transition Matrix A
A = np.matrix([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Define the Control Input Matrix B
B = np.matrix([[(dt**2)/2, 0],
               [0,(dt**2)/2],
               [dt,0],
               [0,dt]])

# Define Measurement Mapping Matrix
H = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0]])

#Initial Process Noise Covariance
Q = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
               [0, (dt**4)/4, 0, (dt**3)/2],
               [(dt**3)/2, 0, dt**2, 0],
               [0, (dt**3)/2, 0, dt**2]]) * std_acc**2

#Initial Measurement Noise Covariance
R = np.matrix([[x_std_meas**2,0],
               [0, y_std_meas**2]])

#Initial Covariance Matrix
P = np.eye(A.shape[1])

def main():

    # Create opencv video capture object
    VideoCap = cv2.VideoCapture(0)
    #Create Arduino Serial Object
    ArduinoData = serial.Serial(port='COM3', baudrate=115200)

    #Create Classifier object
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    x_t_1 = X # state at time t-1
    p_t_1 = P # covariance matrix at time t-1

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
            x_t_1,p_t_1 = predict(A,B,x_t_1,U,p_t_1,Q) #predicted State and Covariance 

            # Draw a rectangle as the predicted object position
            cv2.rectangle(frame, (int(x_t_1[0] - w/2), int(x_t_1[1] - h/2)), (int(x_t_1[0] + w/2), int(x_t_1[1] + h/2)), (0,255, 0), 2)
            cv2.circle(frame, (int(x_t_1[0]),int(x_t_1[1])) , 5,(0,255,0),3)

            # Update

            Z = [[centerx],[centery]] # measurement

            (x_t,p_t) = update(H,p_t_1,R,Z,x_t_1) # Estimated State and Covariance
            
            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (int(x_t[0] - w/2), int(x_t[1] - h/2)), (int(x_t[0] + w/2), int(x_t[1] + h/2)), (0, 0, 255), 2)
            cv2.circle(frame, (int(x_t[0]),int(x_t[1])) , 5,(0,0,255),3)

            cv2.putText(frame, "Estimated Position", (int(x_t[0] + 30), int(x_t[1] + 30)), 0, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (int(x_t_1[0] + 15), int(x_t_1[1])), 0, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Measured Position", (int(centerx + 15), int(centery - 15)), 0, 0.8, (0,191,255), 2)

            # Update prediction value
            x_t_1 = x_t
            p_t_1 = p_t

            #Send Center Data to Arduino
            ArduinoSentData=str(int(width))+','+ str(int(x_t[0]))+'\r'
            # print(ArduinoSentData)
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
