import numpy as np

def predict(A,B,X,U,P,Q):
    #Update State
    #X_t = AX_(t-1) + BU_(t-1)
    X = np.dot(A,X) + np.dot(B,U)

    #Update error Covariance
    #P = APA' + Q
    P = np.dot(np.dot(A, P), A.T) + Q
    return X,P

def update(H,P,R,Z,X):
        # S = H*P*H'+R
        S = H@P@(H.T) + R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = P@(H.T)@(np.linalg.inv(S))

        X = np.round(X + K@(Z-H@X))

        I = np.eye(H.shape[1])

        # Update error covariance matriX
        P = (I - (K@H))@P
        return X,P