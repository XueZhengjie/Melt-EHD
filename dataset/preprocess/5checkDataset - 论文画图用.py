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
def csv_imread(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=np.float32, skip_header=0, skip_footer=0)
    dataNorm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    imgt = cv2.applyColorMap(dataNorm, cv2.COLORMAP_JET)
    return imgt
def ReadCornerPoints(savePath):
    # 再读取
    with open(savePath, 'r') as file:
        reader = csv.reader(file)
        cornerPointsT=[]
        cornerPointsRGB=[]
        for i, row in enumerate(reader):
            if i >= 4 and i <8:
                current_corner_points = cornerPointsRGB
            elif i == 8:
                break
            else:
                current_corner_points = cornerPointsT    
            current_corner_points.append(list(map(float,row)))
    return cornerPointsT,cornerPointsRGB

def WarpPerspectiveImg(img,cornerPoints,dstPoint):
    pts1 = np.float32(cornerPoints)
    pts2 = np.float32([[0, 0], [dstPoint[0], 0], [0, dstPoint[1]], [dstPoint[0], dstPoint[1]]])
    dst = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts1, pts2), dstPoint)
    return dst

def RegisterWithTransform2(csvDir,imgDir,cornerPointsT,cornerPointsRGB,dstPoint):
    
    print("请输入利于分辨对齐的温度图【第几帧】和刚才混合图的最佳img数字（空格隔开）：")
    a,b=map(int,input().split())
    offset=b-a
    csvList=[]
    imgList=[]
    
    fileList=os.listdir(csvDir)
    fileList.sort(key = lambda x: int(x[:-4])) ##文件名按数字排序
    
    for filename in fileList:
        if filename.endswith('.csv'):
           csvList.append(filename)
           imgName=str((int(filename.split('.')[0]))+offset)+'.jpg'
           imgList.append(imgName)
        
    
    return csvList,imgList

def Distance(p1, p2):
    """计算两个点之间的距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def GetMidpoint(p1, p2):
    p=[(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]
    ##第三个参数是离原点距离
    p.append(math.sqrt(p[0]**2 + p[1]**2))    
    return p

def PointsFit(p1,p2,p3=None):
    if p3 is not None:
        x = np.array([p1[0], p2[0], p3[0]])
        y = np.array([p1[1], p2[1], p3[1]]) 
        coefficients = np.polyfit(x, y, 1)
        return coefficients[0]    
    else:
        slope = (p1[1]-p2[1])/(p1[0]-p2[0])
        return slope
    
def calculate_lines_gauss_parameters(MaxLineWidth,Contrast):
    ContrastHigh = Contrast
    ContrastLow = ContrastHigh / 3.0
    if (MaxLineWidth < math.sqrt(3.0)):
        ContrastLow = ContrastLow * MaxLineWidth / math.sqrt(3.0)
        ContrastHigh = ContrastHigh * MaxLineWidth / math.sqrt(3.0)
        MaxLineWidth = math.sqrt(3.0)
    HalfWidth = MaxLineWidth / 2.0
    Sigma = HalfWidth / math.sqrt(3.0)
    Help = -2.0 * HalfWidth / (math.sqrt(6.283185307178) * pow(Sigma,3.0)) * math.exp(-0.5 * pow(HalfWidth / Sigma,2.0))
    High = math.fabs(ContrastHigh * Help)
    Low = math.fabs(ContrastLow * Help)
    return Sigma, Low, High
        
def GetIntersection(imgPath):
    print("######开始计算线和交点######")
    Image=ha.read_image(imgPath)
    # Width, Height = ha.get_image_size(Image)
    #灰度化
    Channels=ha.count_channels (Image)
    if (Channels == 3 or Channels == 4):
        Image=ha.rgb1_to_gray (Image)
        
    #中值滤波
    ImageMedian=ha.median_image(Image, 'circle', 1, 'mirrored')
    # ImageEmphasize=ha.emphasize (ImageMedian, 7, 7, 1)
    # ImageBilateral=ha.bilateral_filter (ImageEmphasize, ImageEmphasize, 20, 20, [], [])

    #求线
    Sigma, Low, High=calculate_lines_gauss_parameters(10, 80) #(4, 90)
    Lines=ha.lines_gauss(ImageMedian, Sigma, Low, High, 'light', 'true', 'bar-shaped', 'true')
    NumberOri=ha.count_obj (Lines)
    UnionContours=ha.union_collinear_contours_xld (Lines, 40, 1, 15, 0.15, 'attr_keep')#(Lines, 5, 0.5, 50, 0.15, 'attr_keep')
    SelectedContours=ha.select_contours_xld (UnionContours, 'contour_length', 15, 2000, -1, -1)
    Number=ha.count_obj (SelectedContours)


    #求交点
    Points=[]
    for i in range(1,Number):
        ObjectSelected1=ha.select_obj (SelectedContours, i)
        for j in range(i+1,Number+1):
            ObjectSelected2=ha.select_obj (SelectedContours, j)
            crossRow,crossColumn,IsOverlapping=ha.intersection_contours_xld(ObjectSelected1, ObjectSelected2, 'all')
            if crossRow != []:
                for t in range(len(crossRow)):
                    Points.append([crossColumn[t],crossRow[t]])
    # print(Points)

    # 显示一下线看看问题
    # WindowHandle = ha.open_window(0, 0, Width[0], Height[0], 
    #                             father_window=0,mode='visible',machine='')
    # ha.disp_obj(Image,WindowHandle)
    # ha.wait_seconds(2)
    # ha.clear_window(WindowHandle)
    # ha.set_color(WindowHandle,'red')
    # ha.disp_obj(SelectedContours, WindowHandle)
    # ha.wait_seconds(50000)

    ##加入离原点距离并按距离排序
    for p in Points:
        p.append(math.sqrt(p[0]**2 + p[1]**2))    
    Points=sorted(Points,key=(lambda x:x[2]))

    ###############合并交点##############
    #NOTE:这里的值可以稍微给大点，避免有很近的点叠在一起看不清。反正识别的点歪了可以手动删补hhh
    mergedPoints=[]
    jumpIndex=[]
    for i in range(len(Points)-1):
        if i in jumpIndex:
            continue
        for j in range(i+1,len(Points)):
            if abs(Points[i][2]-Points[j][2])<15 and Distance(Points[i],Points[j])<20:
                Points[i]=GetMidpoint(Points[i],Points[j])
                jumpIndex.append(j)
        mergedPoints.append(Points[i])
        
    print("line num :"+str(Number)+"\npoints merged from "+str(len(Points))+" to "+str(len(mergedPoints)))    
    return Points,mergedPoints
    '''
    #点的排序及编号问题
    首先我们认为线有几条这个数量是确定的，例如目前是19*19条
    排除最边缘的线，可使用网格线为17*17（how*
    每一条线发生变形也是连续的，从每条横线看，对于提取出的点来说，满足：从左到右123三个点，3离2不超过【网格未变形距离】，12与23的夹角
    '''
    
def ShowIntersectionImg(imgPath,Points,mergedPoints,showAllPoints=False,savePath=None):
    img=cv_imread(imgPath)
    img2=img.copy()

    if showAllPoints:
        for point in Points:
            cross=tuple([round(point[0]),round(point[1])])
            cv2.drawMarker(img,cross,color=(0, 0, 255),markerSize = 10, markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
        cv2.namedWindow('image_draw_marker', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('image_draw_marker', img)

    for point in mergedPoints:
        cross=tuple([round(point[0]),round(point[1])])
        cv2.drawMarker(img2,cross,color=(0, 255, 0),markerSize = 10, markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
    cv2.namedWindow('image_draw_marker2', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('image_draw_marker2', img2)

    # mergedPoints=sorted(mergedPoints,key=(lambda x:x[1]))
    #
    if savePath is not None:
        cv2.imencode('.jpg', img2)[1].tofile(savePath)
        
    # cv2.waitKey(0) #NOTE:想看的话就打开
    cv2.destroyWindow('image_draw_marker2')
    if showAllPoints:
        cv2.destroyWindow('image_draw_marker')
    return img2

def BoxSelectPoints(image,Points,num,savePath=None):
    #读取图像l画出多边形：鼠标取点并显示，以回车作为结束，结束后连线可视化多边形
    polygonPoints=[]# 环形点击
    polyLines=[]
    img=image.copy()
    #选点作为多边形框
    def on_EVENT_BUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            polygonPoints.append([x, y])
            if len(polygonPoints) > 1:
                # 添加一条线段到线段集中
                polyLines.append((polygonPoints[-2], polygonPoints[-1]))
            # cv2.circle(img, (x, y), 2, (255, 0, 0), thickness=2)
            # if len(polygonPoints) > 1:
            #     cv2.line(img, polygonPoints[len(polygonPoints)-1], polygonPoints[len(polygonPoints)-2], (0, 0, 255), 2)
            # cv2.imshow("click to choose the polygonPoints, ENTER to finish", img)
            # Check for key presses  
            # key = cv2.waitKey(1) & 0xFF
            # if key == 13: # Enter key pressed
            #     return
            # elif key == 27: # Escape key pressed
            #     cv2.destroyAllWindows()
            #     exit()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 删除最后一个点以及对应的线段
            if len(polygonPoints) > 0:
                polygonPoints.pop()
                if len(polyLines) > 0:
                    polyLines.pop()
        # 在图像上绘制点集和线段集
        # img_copy = img.copy()
        for pt in polygonPoints:
            cv2.circle(img, pt, 4, (255, 0, 0), -1)
        for line in polyLines:
            cv2.line(img, line[0], line[1], (0, 0, 255), 2)
        cv2.imshow("click to choose the polygonPoints, Lclick to add, Rclick to del, ENTER to finish", img)      
        
              
    cv2.namedWindow("click to choose the polygonPoints, Lclick to add, Rclick to del, ENTER to finish",cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("click to choose the polygonPoints, Lclick to add, Rclick to del, ENTER to finish", on_EVENT_BUTTONDOWN)
    while True:
        cv2.imshow("click to choose the polygonPoints, Lclick to add, Rclick to del, ENTER to finish", img)
        key = cv2.waitKey(1)
        if key == 13:
            break
    
    # cv2.waitKey(0)
    
    if len(polygonPoints) > 1:
    #     cv2.polylines(img, [np.array(polygonPoints)], True, (0, 0, 255),3)         
        cv2.line(img, polygonPoints[0], polygonPoints[len(polygonPoints)-1], (0, 0, 255), 2)
    cv2.imshow("click to choose the polygonPoints, Lclick to add, Rclick to del, ENTER to finish", img)
    cv2.waitKey(0)
    
    #剔除多边形以外的点
    # Create a binary mask of the polygon
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygonPoints)], 255)
    # Create a new list of points within the polygon
    innerPoints = []
    for point in Points:
        if cv2.pointPolygonTest(np.array(polygonPoints), (point[0],point[1]), False) >= 0:
            innerPoints.append(point)
    for point in innerPoints:
        cv2.circle(img, (round(point[0]),round(point[1])), 2, (255, 0, 0), 1)
    cv2.imshow("click to choose the polygonPoints, Lclick to add, Rclick to del, ENTER to finish", img)
    # cv2.waitKey(0)
    print("共",str(len(innerPoints)),"个点，需要",str(num),"行列共",str(num*num),"个点")
    # print("\n点的坐标为:",innerPoints)   
    cv2.destroyWindow("click to choose the polygonPoints, Lclick to add, Rclick to del, ENTER to finish")
    if len(innerPoints)!=num*num:
        print("少点了，开始手动删补点，左键加点，右键删除最近点")
        print("差",str(num*num-len(innerPoints)),"个点：目前共",str(len(innerPoints)),"个点，需要",str(num),"行列共",str(num*num),"个点")
    else:
        print("点已经找全！看看有没歪哈，左键加点，右键删除最近点")    
    # 定义鼠标回调函数
    def MouseCallback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 添加一个点到点集中
            innerPoints.append([x, y])
            print("差",str(num*num-len(innerPoints)),"个点：目前共",str(len(innerPoints)),"个点，需要",str(num),"行列共",str(num*num),"个点")
            if len(innerPoints)==num*num:
                print("点已经找全！")  
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 删除距离点击点最近的点
            if len(innerPoints) > 0:
                distances = [((x - pt[0]) ** 2 + (y - pt[1]) ** 2) for pt in innerPoints]
                idx_to_remove = np.argmin(distances)
                innerPoints.pop(idx_to_remove)
            print("差",str(num*num-len(innerPoints)),"个点：目前共",str(len(innerPoints)),"个点，需要",str(num),"行列共",str(num*num),"个点")
        # 在图像上绘制点集
        img_copy = image.copy()
        cv2.polylines(img_copy, [np.array(polygonPoints)], True, (0, 0, 255),3)     
        for point in innerPoints:
            cv2.circle(img_copy, (round(point[0]),round(point[1])), 3, (255, 0, 0), 2)
        cv2.imshow('Left click add, right click delete, ENTER to finish', img_copy)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 13 & len(innerPoints)==num*num: # ENTER key pressed 并且num*num个点 NOTE！
            return 
        elif key == 27: # Escape key pressed
            cv2.destroyAllWindows()
            exit()
                
    cv2.namedWindow("Left click add, right click delete, ENTER to finish",cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("Left click add, right click delete, ENTER to finish", MouseCallback)
    # cv2.imshow("Left click add, right click delete, f to finish", img2)
    cv2.waitKey(0)
    cv2.destroyWindow("Left click add, right click delete, ENTER to finish")
    
    return innerPoints
    
def NumberPoints(image,points,num,savePath=None):
    point_size=5
    img=image.copy()
    # 取最左边n个点 按y排序得到每列第一个点
    points.sort(key=lambda x: x[0])
    leftmost_points = points[:num]
    points=points[num:]
    leftmost_points.sort(key=lambda x: x[1])
    sorted_points=[[] for i in range(num)]
    # Draw the blue points on the image and display them with their index
    for i, point in enumerate(leftmost_points):
        cv2.circle(img, (int(point[0]), int(point[1])), point_size, (255,0,0), -1)
        cv2.putText(img, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        # Add the point to a separate list
        sorted_points[i].append(point)
        
    # 对于一行点，对第一个点向右按照距离and斜率（angle1）取第二个点，然后按照前两点方向的距离+向量夹角（angle2）取第三个点，
    # 后续点用前三个点拟合直线取k+距离+两点与k的夹角（angle3），直到满足n个点
    angle1=15
    angle2=20
    angle3=15
    cv2.namedWindow("Number the points, press to goon, click to add", cv2.WINDOW_AUTOSIZE)
    for i in range(num):
        cv2.imshow("Number the points, press to goon, click to add", img)
        closest_point = None
        closest_distance = float('inf')
        first_point=sorted_points[i][0]
        for point in points:
            #可以用距离范围初筛一下加速
            if point in sorted_points[i]:#(point[0]-first_point[0])==0:#
                continue
            slope = abs(PointsFit(point,first_point))
            if point[0] > first_point[0] and slope <= np.tan(np.deg2rad(angle1)):
                # 计算距离找最近点
                distance = Distance(point,first_point)
                if distance < closest_distance:
                    closest_point = point
                    closest_distance = distance
        sorted_points[i].append(closest_point)   
        points.remove(closest_point)         
        cv2.circle(img, (int(sorted_points[i][1][0]), int(sorted_points[i][1][1])), point_size, (0,0,255), -1)     

        # 然后按照前两点方向的距离+向量夹角angle2取第三个点，
        closest_point = None
        closest_distance = float('inf')
        for point in points:
            if point in sorted_points[i]:#(point[0]-sorted_points[i][0])==0:
                continue
            #计算向量夹角
            #可以用距离范围初筛一下加速?
            v1=[sorted_points[i][1][0]-first_point[0],sorted_points[i][1][1]-first_point[1]]
            v2=[point[0]-sorted_points[i][1][0],point[1]-sorted_points[i][1][1]]
            cos_angle=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            angle=np.rad2deg(np.arccos(cos_angle))
            if point[0] > sorted_points[i][1][0] and abs(angle)<=angle2:
                #计算距离找最近点
                distance=Distance(sorted_points[i][1],point)
                if distance<closest_distance:
                    closest_point=point
                    closest_distance=distance
        sorted_points[i].append(closest_point)
        points.remove(closest_point) 
        cv2.circle(img, (int(sorted_points[i][2][0]), int(sorted_points[i][2][1])), point_size, (0,255,0), -1)  
        
        #后续点用前三个点拟合直线取k+距离+两点与k的夹角，直到满足n个点
        while len(sorted_points[i]) >= 3 and len(sorted_points[i]) <num:
            last_point = sorted_points[i][-1]
            closest_point = None
            closest_distance = float('inf')
            for point in points:
                if (point[0]-last_point[0])==0:#point in sorted_points[i]:#这里可能有问题
                    continue
                # Calculate the vector k using the last 3 points in the current row
                k1 = PointsFit(sorted_points[i][-1],sorted_points[i][-2],sorted_points[i][-3])
                k2 = PointsFit(point,last_point)
                # Calculate the angle between vectors k1+2
                angle = math.fabs(np.arctan((k1-k2)/(float(1 + k1*k2)))*180/np.pi)
                # Check if the angle is less than 20 degrees and the distance is less than the previous closest distance
                if angle <= angle3 and point[0] > sorted_points[i][-1][0] :
                    distance=Distance(last_point, point)
                    if distance < closest_distance:
                        closest_point = point
                        closest_distance = distance
            if closest_point is not None:
                sorted_points[i].append(closest_point)
                points.remove(closest_point) 
                cv2.circle(img, (int(closest_point[0]), int(closest_point[1])), point_size, (255,255,0), -1)          
            else:
                break
        # print(len(points))
        cv2.imshow("Number the points, press to goon, click to add", img)
        while len(sorted_points[i]) <num:
            print("第",str(i+1),"行漏点啦！请左键点击补点")        
            #点击取点，判断points里剩下点中与所取点最近的点，加入sorted_points[i]直到len(sorted_points[i])==14,对sorted_points[i]按照x值排序
            def mouse_add_sorted_point(event, x, y, flags, params):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if len(sorted_points[i]) < num:
                        closest_point = None
                        closest_distance = float('inf')
                        for point in points:
                            if point in sorted_points[i]:
                                continue
                            distance = Distance([x,y], point)
                            if distance < closest_distance:
                                closest_point = point
                                closest_distance = distance
                        # Check if a closest point was found
                        if closest_point is not None:
                            sorted_points[i].append(closest_point)
                            cv2.circle(img, (int(closest_point[0]), int(closest_point[1])), point_size, (0,255,255), -1)
                            cv2.imshow("Number the points, press to goon, click to add", img)
                            if len(sorted_points[i]) == num:
                                sorted_points[i].sort(key=lambda x: x[0])
                                return
            cv2.setMouseCallback("Number the points, press to goon, click to add", mouse_add_sorted_point)
            cv2.imshow("Number the points, press to goon, click to add", img)
            cv2.waitKey(0)
        cv2.waitKey(0)

    sorted_points_all=[]
    for i in range(num):
        for j in range(len(sorted_points[i])):
            sorted_points_all.append(sorted_points[i][j])
    for i, point in enumerate(sorted_points_all):
        cv2.putText(img, str(i+1), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow("Number the points, press to goon, click to add", img)    
    cv2.waitKey(0)
    cv2.destroyWindow("Number the points, press to goon, click to add")
    return sorted_points_all

def GetCSVTemperature(csvPath,image,dstPoint,cornerPointsT,points,savePath=None):
    #读取csv
    
    data = np.genfromtxt(csvPath, delimiter=',', dtype=np.float32, skip_header=0, skip_footer=0)
    dataNorm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = cv2.applyColorMap(dataNorm, cv2.COLORMAP_JET)
    # cv2.imshow('Temperature Map', img)
    # cv2.waitKey(0)
    
    pts1 = np.float32(cornerPointsT)
    pts2 = np.float32([[0, 0], [dstPoint[0], 0], [0, dstPoint[1]], [dstPoint[0], dstPoint[1]]])
    M=cv2.getPerspectiveTransform(pts2, pts1)
    
    for i,point in enumerate(points):
        points[i]=(point[0],point[1])
    points=np.float32(points)
    pointsT = cv2.perspectiveTransform(points.reshape(-1, 1, 2), M)
    # print(np.shape(pointsT))
    pointsT=pointsT.reshape(np.shape(pointsT)[0],np.shape(pointsT)[2])
    # print(np.shape(pointsT))
    # cv2.destroyWindow('Points map')
    
    #拿点周围5*5范围的最大值!!!奇数哈！
    areaSize=13 #奇数
    t=[]
    for i,point in enumerate(pointsT):
        center=(round(point[0]), round(point[1]))
        dataGet = data[center[1]-int(areaSize/2):center[1]+int(areaSize/2)+1, center[0]-int(areaSize/2):center[0]+int(areaSize/2)+1]
        dataMax = np.max(dataGet)
        t.append([dataMax])
        ##可视化提取区域啦
        for i in range(center[0]-int(areaSize/2),center[0]+int(areaSize/2)+1):
            for j in range(center[1]-int(areaSize/2),center[1]+int(areaSize/2)+1):
                img[j,i] = [0,255,0]#注意这里是img的行列坐标而不是xy
    t=np.float32(t)    
    points = np.hstack((points, t))        

    #look look pointsT 看去了哪些范围的pointsT
    for point in pointsT:
        cv2.circle(img, (int(point[0]), int(point[1])), 1, (0,0,255), -1)
        
    cv2.namedWindow("PointsT map",cv2.WINDOW_NORMAL)    
    cv2.imshow("PointsT map", img)
    
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 4, (255,0,0), -1)
        cv2.putText(image, str(format(point[2], '.1f')), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imshow("Points2 map", image)
    
    cv2.waitKey(0)
    # cv2.destroyWindow("PointsT map")
    cv2.destroyWindow("Points2 map")
    
    return points


def SavePoints(points,savePath):
    # 写入csv文件保存
    with open(savePath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(points)
    print("点已保存到:",savePath)

def ReadRecoeder(txtSaver,txtPath):
    with open(txtPath, 'a') as fd:
        for d in txtSaver:
            fd.writelines(str(d))
            fd.writelines("\n")   
            
def ReadPoints(savePath):
    # 再读取
    with open(savePath, 'r') as file:
        reader = csv.reader(file)
        points=[]
        for row in reader:
            points.append(list(map(float,row)))
    return points

def ShowPoints(savePath,points):
    img=cv_imread(savePath)
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 4, (255,0,0), -1)
    cv2.imshow('Points map', img)
    cv2.waitKey(0)
    cv2.destroyWindow('Points map')

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
tempDir=os.path.join(mainDir,configs['tempDir'])
savePointsDir=os.path.join(mainDir,configs['savePointsDir'])
txtPath=os.path.join(mainDir,configs['txtPath'])
dstPoint = configs['dstPoint']  # 变换目标大小
dstPoint = (dstPoint,dstPoint)
n=configs['n'] #点阵行列数
x=configs['x']

#改给check用的
cornerPath="C:\\Users\\1\\Desktop\\Research relation\\3d打印合作\\论文\\画图\\origin\\pointsT\\corner_points.csv"
csvPath="C:\\Users\\1\\Desktop\\Research relation\\3d打印合作\\论文\\画图\\origin\\pointsT\\csv"
savePointsPath="C:\\Users\\1\\Desktop\\Research relation\\3d打印合作\\论文\\画图\\origin\\pointsT\\points"
pointsTmapSavePath="C:\\Users\\1\\Desktop\\Research relation\\3d打印合作\\论文\\画图\\origin\\Tmap with points.jpg"
TmapSavePath="C:\\Users\\1\\Desktop\\Research relation\\3d打印合作\\论文\\画图\\origin\\Tmap.jpg"
#开始检查数据
de=False
de_name="756.csv"
csv_files = [f for f in os.listdir(savePointsDir) if f.endswith('.csv')]
# 将文件名转换为整数并排序
# csv_files = sorted(csv_files, key=lambda x: int(os.path.splitext(x)[0]))

cornerPointsT,cornerPointsRGB=ReadCornerPoints(cornerPath)
start_index = 0
jump = 1
if de:
    start_index = max(0,csv_files.index(de_name)-10)
    jump = 1
for i, csv_file in enumerate(csv_files[start_index:]):
    if i % jump == 0 and os.path.exists(os.path.join(csvDir, csv_file)):  # 每隔5个文件取一个文件
        
        points=ReadPoints(os.path.join(savePointsDir, csv_file))
        
        data = np.genfromtxt(os.path.join(csvDir, csv_file), delimiter=',', dtype=np.float32, skip_header=0, skip_footer=0)
        dataNorm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = cv2.applyColorMap(dataNorm, cv2.COLORMAP_JET)
        # cv2.imshow('Temperature Map', img)
        # cv2.waitKey(0)
        cv2.imencode('.jpg',img)[1].tofile(TmapSavePath)
        pts1 = np.float32(cornerPointsT)
        pts2 = np.float32([[0, 0], [dstPoint[0], 0], [0, dstPoint[1]], [dstPoint[0], dstPoint[1]]])
        M=cv2.getPerspectiveTransform(pts2, pts1)
        
        for i,point in enumerate(points):
            points[i]=(point[0],point[1])
        points=np.float32(points)
        pointsT = cv2.perspectiveTransform(points.reshape(-1, 1, 2), M)
        # print(np.shape(pointsT))
        pointsT=pointsT.reshape(np.shape(pointsT)[0],np.shape(pointsT)[2])
        # print(np.shape(pointsT))
        # cv2.destroyWindow('Points map')
        
        #拿点周围5*5范围的最大值!!!奇数哈！
        areaSize=13 #奇数
        t=[]
        for i,point in enumerate(pointsT):
            center=(round(point[0]), round(point[1]))
            dataGet = data[center[1]-int(areaSize/2):center[1]+int(areaSize/2)+1, center[0]-int(areaSize/2):center[0]+int(areaSize/2)+1]
            dataMax = np.max(dataGet)
            t.append([dataMax])
            ##可视化提取区域啦
            # for i in range(center[0]-int(areaSize/2),center[0]+int(areaSize/2)+1):
            #     for j in range(center[1]-int(areaSize/2),center[1]+int(areaSize/2)+1):
                    # img[j,i] = [0,255,0]#注意这里是img的行列坐标而不是xy
        t=np.float32(t)    
        points = np.hstack((points, t))        

        #look look pointsT 看去了哪些范围的pointsT
        for point in pointsT:
            cv2.circle(img, (int(point[0]), int(point[1])), 4, (0,255,255), -1)
            
        cv2.namedWindow("PointsT map:"+csv_file,cv2.WINDOW_NORMAL)    
        cv2.resizeWindow("PointsT map:"+csv_file, 1200, 1000)
        cv2.imshow("PointsT map:"+csv_file, img)
        cv2.imencode('.jpg',img)[1].tofile(pointsTmapSavePath)

        key = cv2.waitKey(0) & 0xFF
        if key == 27: # Escape key pressed
            cv2.destroyAllWindows()
            exit()
        else :
            cv2.destroyAllWindows()
    else:
        # 如果不存在，则继续下一个文件
        continue

 
print("###############当前工况已完成################")
    
    
    
    
    
