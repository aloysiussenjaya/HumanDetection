import cv2
import numpy as np
import socket
import time


# Define IP Address for Arduinos and PORT Number
# Arduino_1 = '192.168.100.21'
# Arduino_2 = '192.168.100.20'

#Arduino_1 = '172.20.10.13'
#Arduino_2 = '172.20.10.14'

## Punya Joi 
Arduino_1 = '192.168.43.137'
Arduino_2 = '192.168.43.87'

PORT_1 = 8888
MONITORING_PORT = 4500

class Connection:
    def __init__(self, HOST, PORT):
        self.HOST = HOST
        self.PORT = PORT
        
    def createSocket(self):
        return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def connect(self, sock):
        server_address = (self.HOST, self.PORT)
        print('connecting to {} port {}'.format(*server_address))
        sock.connect(server_address)
    
    def sendPacket(self, message, sock):
        print('sending {!r}'.format(message))
        sock.sendall(message)
    
    def closeSocket(self, sock):
        print('closing socket')
        sock.close()


def clientSocket(msg, lamp):
    if lamp == 1:
        tcpConnection = Connection(Arduino_1, PORT_1)
    elif lamp == 2:
        tcpConnection = Connection(Arduino_2, PORT_1)
    else:
        tcpConnection = Connection(Server_Result, MONITORING_PORT)
    sock = tcpConnection.createSocket()
    tcpConnection.connect(sock)

    try:
        
        # Define
        msg = msg
        tcpConnection.sendPacket(msg, sock)
        
        ###### Look for the responses from Arduino 
        #amount_received = 0
        #amount_expected = len(msg)

        #while amount_received < amount_expected:
        #    data = sock.recv(200)
        #    amount_received += len(data)
        #    print('received {!r}'.format(data))
        
    finally:
        tcpConnection.closeSocket(sock)

        
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# human height sample 175cm
known_distance = 150 #centimeter
known_frame_height = 388 # 388 # 1164
real_sample_height = 167

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getFocalLength(known_distance, real_sample_height, known_frame_height):
    focal_length = (known_frame_height * known_distance) / real_sample_height
    return focal_length

def calculateDistance(Focal_length, real_sample_height, real_time_frame_height):
    result = (real_sample_height * Focal_length) / real_time_frame_height
    return result
    
def getObjects(img, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float, confs))

    if len(objects) == 0:
        objects = classNames

    objectInfo = []

    #if(len(confs) > 0):
    #    print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    #print(indices)

    for i in indices:
        i = i[0]
        box = bbox[i]
        prob = confs[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        
        #print("\n\nHEIGHT : %s" %box[3])
        
        #clientSocket(b'%s'%box[3], 0)
        #if(box[3] < 220 && box[3] < 180):
        #    clientSocket(b'0', 1)
        #    clientSocket(b'1', 2)
        #else:
        #    clientSocket(b'0', 2)
        #    clientSocket(b'1', 1)
        className = classNames[classIds[i][0]-1]
        if className in objects:
            #if prob > 0.65:
            objectInfo.append([[x,y,w,h], className])
            focal_length = getFocalLength(known_distance, real_sample_height, known_frame_height)
            Distance = calculateDistance(focal_length, real_sample_height, h)
            if(draw):
                cv2.rectangle(img, (x,y), (x+w,y+h), color=(0,225,0), thickness=2)
                #cv2.putText(img, f"Distance = {round(Distance,2)} CM", (box[0]+7, box[1]+30), cv2.FONT_HERSHEY_COMPLEX,(0.2*box[3])/80, (255,0,0),1)
                #cv2.putText(img, f"Distance = {round(Distance,2)} CM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2)
                #cv2.putText(img, f"Person : {round(prob*100,2)} %", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2)
                #cv2.putText(img, className.upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img, objectInfo
    #if len(classIds) != 0:
    #    for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    #        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    #        cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
    #                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
    #                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3,640) # 1920 x 1440
    cap.set(4,480)
    cap.set(5,60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    cap.set(10,60)
    #cap.set(11,50)
    #cap.set(12,0)
    #cap.set(13,500)
    #cap.set(14,-100)
    #cap.set(15, -0.8)
    #cap.set(16,30)
    #print(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        success,img = cap.read()
        result, objectInfo = getObjects(img,objects=['person'])
        if not objectInfo:
            clientSocket(b'1', 1)
        else :
            for final_box in objectInfo:
                height = final_box[0][3]
                print("Person Height :", height)
                print(height)
                if height != 0:
                    focal_length = getFocalLength(known_distance, real_sample_height, known_frame_height)
                    Distance = calculateDistance(focal_length, real_sample_height, height) + 25
                    print("Distance :", round(Distance,2), "cm")
                    if (Distance > 149.9) and (Distance < 249.9):
                    #clientSocket(b'0', 1)
                        clientSocket(b'0', 1)
                        time.sleep(0.3)
                    elif (Distance > 349.9) and (Distance < 449.9):
                        clientSocket(b'0', 2)
                        time.sleep(0.3)
                    else:
                        clientSocket(b'1', 1)
                        clientSocket(b'1', 2)
                        time.sleep(0.3)
                        
                else:
                    clientSocket(b'1', 1)
                    clientSocket(b'1', 2)
                    time.sleep(0.3)
    #             elif height > 350:
    #                 clientSocket(b'1', 1)
    #                 clientSocket(b'0', 2)
    #             else:
    #                 clientSocket(b'1', 1)
    #                 clientSocket(b'1', 2)
                    
    #         print(objectInfo)
        
            
        cv2.imshow("Output",img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break