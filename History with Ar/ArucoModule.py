import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadAugImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    # print(noOfMarkers)
    augDict = {}
    for iP in myList:
        key = int(os.path.splitext(iP)[0])
        val = cv2.imread(f'{path}/{iP}')
        augDict[key] = val

    return augDict



def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    # arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    arucoParam = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict,arucoParam)
    bboxs, ids, rejected = detector.detectMarkers(imgGray)
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)

    return [bboxs,ids]

def augmentAruco(bbox,id,img,imgAug, drawId=True):
    tl = bbox[0][0][0] , bbox[0][0][1]
    tr = bbox[0][1][0] , bbox[0][1][1]
    br = bbox[0][2][0] , bbox[0][2][1]
    bl = bbox[0][3][0] , bbox[0][3][1]

    h, w, c = imgAug.shape
    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix, _ =cv2.findHomography(pts2,pts1)
    imgOut = cv2.warpPerspective(imgAug,matrix,(img.shape[1],img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int),(0,0,0))
    imgOut = img + imgOut

    # if drawId:
    #     cv2.putText(imgOut, str(id), (int(tl[0]),int(tl[1])) ,cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    
    return imgOut
    

def main():
    cap = cv2.VideoCapture(0)
    augDic = loadAugImages("Markers")

    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)

        #checking markers found
        if len(arucoFound[0])!=0:
            for box,id in zip(arucoFound[0],arucoFound[1]):
                if int(id) in augDic.keys():
                    img = augmentAruco(box, id, img, augDic[int(id)])

        cv2.imshow("Image",img)
        cv2.waitKey(1)     

if __name__=="__main__":
    main()

