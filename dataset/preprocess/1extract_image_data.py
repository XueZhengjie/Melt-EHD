# coding=utf-8

# 手动保存csv文件，每个工况一个文件夹，下分csv image
# 手动：csv从合适的帧开始存，存到认为不需要的帧了就行了
# 运行这个提取image所有帧
# image对齐后会自动计算每一帧csv对应的图片

import os
import cv2
import shutil
import time
import yaml

config = "config.yaml"
with open(config, 'r', encoding='utf-8') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)

mainDir=configs['mainDir']    
imgDir=os.path.join(mainDir,configs['imgDir'])

VIDEO_PATH = configs['VIDEO_PATH']  # 视频地址
EXTRACT_FOLDER = imgDir  # 存放帧图片的位置
EXTRACT_FREQUENCY = 1  # 帧提取频率

def MkDir(Path):
    for path in Path:
        if(os.path.exists(path) == False):
            os.makedirs(path)
# 主操作
def extract_frames(video_path, dst_folder, index):
    # 实例化视频对象
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    # 循环遍历视频中的所有帧
    while True:
        # 逐帧读取
        _, frame = video.read()
        if frame is None:
            break
        # 按照设置的频率保存图片
        if frame_count % EXTRACT_FREQUENCY == 0:
            # 设置保存文件名
            save_path = "{}/{}.jpg".format(dst_folder, index)
            # 保存图片
            cv2.imwrite(save_path, frame)
            index += 1  # 保存图片数＋1
        frame_count += 1  # 读取视频帧数＋1

    # 视频总帧数
    print(f'the number of frames: {frame_count}')
    # 打印出所提取图片的总数
    print("Totally save {:d} imgs".format(index - 1))

    # 计算FPS 方法一 get()
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')  # Find OpenCV version
    # (major_ver, minor_ver, subminor_ver) = (4, 5, 4)
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)  # 获取当前版本opencv的FPS
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)  # 获取当前版本opencv的FPS
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # 计算FPS 方法二 手动计算 总帧数 / 总时间
    # new_vid = cv2.VideoCapture(video_path)
    # start = time.time()  # 开始时间
    # c = 0
    # while True:
    #     _, frames = new_vid.read()
    #     if frames is None:
    #         break
    #     c += 1
    # end = time.time()  # 结束时间
    # fps2 = c / (end - start)  # 总帧数 / 总时间
    # print(f'frames:{c}')
    # print(f'seconds:{end - start}')
    # print("Frames per second using frames / seconds : {0}".format(fps2))
    # new_vid.release()

    # 最后释放掉实例化的视频
    video.release()


def main():
    # 递归删除之前存放帧图片的文件夹，并新建一个
    try:
        shutil.rmtree(EXTRACT_FOLDER)
    except OSError:
        pass
    os.mkdir(EXTRACT_FOLDER)
    # 抽取帧图片，并保存到指定路径
    extract_frames(VIDEO_PATH, EXTRACT_FOLDER, 1)
    print("图片提取完成")

if __name__ == '__main__':
    main()
