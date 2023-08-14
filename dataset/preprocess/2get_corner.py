import halcon as ha
import numpy as np
import cv2
import math
import pandas as pd
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import yaml

#cv读取中文路径
def cv_imread(file_path):    
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)    
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def SavePoints(points,savePath):
    # 写入csv文件保存
    with open(savePath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(points)
    print("点保存在:",savePath)
    
def WarpPerspectiveCorner(img,dstPoint):

    cornerPoints=[]# 左上，右上，左下，右下顺序点击
    # dst = np.ones(shape=[dstPoint[0],dstPoint[1],3], dtype=np.uint8)
    global dst
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            cv2.circle(img, (x, y), 5, (255, 0, 0), thickness=5)
            # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,5.0, (0, 0, 0), thickness=4)
            cv2.imshow("points_for_warpPerspective:upper left,upper right,lower left, lower right", img)
            cornerPoints.append([x,y])
            if len(cornerPoints) == 4:
                pts1 = np.float32(cornerPoints)
                pts2 = np.float32([[0, 0], [dstPoint[0], 0], [0, dstPoint[1]], [dstPoint[0], dstPoint[1]]])

                global dst
                dst = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts1, pts2), dstPoint)
                cv2.namedWindow("warpPerspective_img", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("warpPerspective_img", dst)

                return 

    cv2.namedWindow("points_for_warpPerspective:upper left,upper right,lower left, lower right",cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("points_for_warpPerspective:upper left,upper right,lower left, lower right", on_EVENT_LBUTTONDOWN)
    cv2.imshow("points_for_warpPerspective:upper left,upper right,lower left, lower right", img)
    cv2.waitKey(0)
    cv2.destroyWindow("points_for_warpPerspective:upper left,upper right,lower left, lower right")
    cv2.destroyWindow("warpPerspective_img")

    # while True:
    #     cv2.imshow("image", img)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27: #or len(cornerPoints) == 4:
    #         break
    print("The four clicked corner points are:", cornerPoints)

    return cornerPoints

def WarpPerspectiveImg(img,cornerPoints,dstPoint):
    pts1 = np.float32(cornerPoints)
    pts2 = np.float32([[0, 0], [dstPoint[0], 0], [0, dstPoint[1]], [dstPoint[0], dstPoint[1]]])
    dst = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts1, pts2), dstPoint)
    return dst

#看exe手动选一帧变化大的csv
def getCorner(csvDir,imgDir,dstPoint):
    
    print("请输入角点清楚的温度图【第几帧】,RGB图【第几帧（这个随意】，输入示例:30 30")
    a,b=map(int,input().split())
    # a=int(input())
    csvPath=csvDir+'/'+str(a)+".csv"
    # Load CSV file
    data = np.genfromtxt(csvPath, delimiter=',', dtype=np.float32, skip_header=0, skip_footer=0)
    # Normalize data to 0-255 range
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Create image from data
    imgt = cv2.applyColorMap(data, cv2.COLORMAP_JET)
    #对t图变换
    cornerPointsT=WarpPerspectiveCorner(imgt,dstPoint)
    # imgt=WarpPerspectiveImg(imgt,cornerPointsT,dstPoint)
    
    # print("请输入好看的RGB图【第几帧】")
    # b=int(input())
    imgPath=imgDir+'/'+str(b)+".jpg"
    img = cv_imread(imgPath)
    cornerPointsRGB=WarpPerspectiveCorner(img,dstPoint)
    
    return cornerPointsT,cornerPointsRGB
    
def MkDir(Path):
    for path in Path:
        if(os.path.exists(path) == False):
            os.makedirs(path)
            
            
config = "config.yaml"
with open(config, 'r', encoding='utf-8') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
    
mainDir=configs['mainDir']    
imgDir=os.path.join(mainDir,configs['imgDir'])

cornerPath=configs['cornerPath']
csvDir=os.path.join(mainDir,configs['csvDir'])

dstPoint = configs['dstPoint']  # 变换目标大小
dstPoint = (dstPoint,dstPoint)
n=configs['n'] #点阵行列数

MkDir([imgDir,csvDir])

# #取四个点做透视变换得到网的框并保存图片
# dst=WarpPerspectiveImg(imgPath,dstPoint)
# cv2.imencode('.jpg', dst)[1].tofile(tempPath)  # 输出图像
cornerPointsT,cornerPointsRGB=getCorner(csvDir,imgDir,dstPoint)
SavePoints(cornerPointsT+cornerPointsRGB,cornerPath)

