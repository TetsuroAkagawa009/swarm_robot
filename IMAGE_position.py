# -*- coding: utf-8 -*-
#カメラ画像からロボットの位置特定（Python）
# 
import datetime
import sys
#sys.path.append('/home/mclab/.local/lib/python3.5/site-packages')
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import os
import pyrealsense2 as rs
from math import atan2, cos, sin, sqrt, pi

stepCount = 0
ImageBox = []

#################
#画像のトリミング
#################
def Image_Cut(img):
    Height = 480 #カット後の高さ
    Width = 640  #カット後の幅
    HeightStart = 0 #カットする開始地点
    WidthStart = 0  #カットする開始地点
    CutHeight = Height - HeightStart
    CutWidth = Width - WidthStart
    # 画像を切り抜き(ここだけは縦、横)
    img = img[HeightStart:HeightStart+Height,WidthStart:WidthStart+Width]
    img = cv2.resize(img, (CutWidth, CutHeight))
    return img

#################
#画像の保存
# rgb:画像
# SaveNum:保存先のディレクトリ名(Image/data「数字」)
#################
def Image_Save(rgb, SaveNum):
  # numpy配列化
  rgb = np.array(rgb,dtype="float32")
  # 画像ファイルを保存先(ディレクトリ)を生成
  DirName = str(int(SaveNum))
  DirName = "Image/data"+str(DirName)
  # もしすでにディレクトリがあるなら
  if(os.path.isdir(DirName) != True):
      os.makedirs(DirName)
      print("RGBフォルダ{0}を作成しました。\n".format(DirName))
  else:
      print("既存RGBフォルダ{0}に上書きします。\n".format(DirName))
  # 画像を保存
  for SaveNumber in range(0, len(rgb)):
    NameRGB = "Image/data{:.0f}/{:.3f}.png".format(SaveNum,SaveNumber)
    cv2.imwrite(NameRGB, rgb[SaveNumber])

## 角度を設定するメソッド
def IMAGE_GetAng(img):
    # 輪郭を決定
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, pts in enumerate(contours):
      # つかう領域を設定
      area = cv2.contourArea(pts)
      # 大きすぎるのと小さすぎるのは無視
      if area < 1e2 or 1e5 < area:
        continue

    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # 主成分分析(PCA)にて方向情報を絞る
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # 先ほどのベクトル情報をもとに角度を計算
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
    # rad -> deg
    angle = int(np.rad2deg(angle))
    print("  Rotation Angle: " + str(angle) + " degrees") # test oprint

    return angle

def IMAGE_GetPosi_GetAng(InputImage):
  kernel = np.ones((3, 3),np.uint8)
  robot1_posi = np.ones((2),np.uint8)
  robot2_posi = np.ones((2),np.uint8)
  robot3_posi = np.ones((2),np.uint8)
  robot1_ang = 0
  robot2_ang = 0
  robot3_ang = 0
  # HSV色空間に変換
  hsv = cv2.cvtColor(InputImage, cv2.COLOR_BGR2HSV)
  # robot1のHSVの値域1  rgb(132,90,46) -> hsv(30, 65, 51) #825a2e
  hsv_min_1type = np.array([0,150,100]) 
  hsv_max_1type = np.array([70,170,365])
  mask1 = cv2.inRange(hsv, hsv_min_1type, hsv_max_1type)
  cv2.imwrite("robot1.png", mask1) # test print
  # robot2のHSVの値域2  rgb(10,111,71) -> hsv(156, 90, 43)  #0a6f47
  hsv_min_2type = np.array([70,200,10])
  hsv_max_2type = np.array([80,250,130])
  mask2 = cv2.inRange(hsv, hsv_min_2type, hsv_max_2type)
  cv2.imwrite("robot2.png", mask2) # test print
  # robot3のHSVの値域3  rgb(8,94,103) -> hsv(185, 92, 40)  #0a6f47
  hsv_min_3type = np.array([90,200,10])
  hsv_max_3type = np.array([200,250,130])
  mask3 = cv2.inRange(hsv, hsv_min_3type, hsv_max_3type)
  cv2.imwrite("robot3.png", mask3) # test print
  # モフォロジー変換
  opening1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
  opening2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
  opening3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel)
  cv2.imwrite("robot1_op.png", opening1) # test print
  cv2.imwrite("robot2_op.png", opening2) # test print
  cv2.imwrite("robot3_op.png", opening3) # test print
  
  mu1 = cv2.moments(opening1, False)
  mu2 = cv2.moments(opening2, False)
  mu3 = cv2.moments(opening3, False)
  #重心から座標取得
  # robot1を見つけたとき
  if (mu1["m00"] != 0):
    robot1_posi = [int(mu1["m10"]/mu1["m00"]) , int(mu1["m01"]/mu1["m00"])]
  # robot2を見つけたとき
  if (mu2["m00"] != 0):
    robot2_posi = [int(mu2["m10"]/mu2["m00"]) , int(mu2["m01"]/mu2["m00"])]
  # robot2を見つけたとき
  if (mu3["m00"] != 0):
    robot3_posi = [int(mu3["m10"]/mu3["m00"]) , int(mu3["m01"]/mu3["m00"])]
    
  # set angle (deg)
  robot1_ang = IMAGE_GetAng(opening1)
  robot2_ang = IMAGE_GetAng(opening2)
  robot3_ang = IMAGE_GetAng(opening3)
  robot1_posi.append(robot1_ang)
  robot2_posi.append(robot2_ang)
  robot3_posi.append(robot3_ang)

  print(robot1_posi, robot2_posi, robot3_posi) # test print
  return robot1_posi, robot2_posi, robot3_posi

def IMAGE_Get(pipe, align):
  # 画像データ取得
  ####################################
  # Wait for a coherent pair of frames: depth and color
  ####################################
  frames = pipe.wait_for_frames()
  aligned_frames = align.process(frames)
  color_frame = aligned_frames.get_color_frame()
  depth_frame = aligned_frames.get_depth_frame()
  #撮れるまで何度も繰り返す
  while not depth_frame or not color_frame:
    frames = pipe.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
  #imageをnumpy arrayに
  color_image = np.asanyarray(color_frame.get_data())
  depth_image = np.asanyarray(depth_frame.get_data())
  cv2.imwrite("test_rgb.png", color_image) # test print
  cv2.imwrite("test_depth.png", depth_image) # test print
  
  # depth image not use for this task
  return color_image

def IMAGE_Init():
  stepCount = 0
  ImageBox = []
  ##############################################################
  #      Real sense の初期処理
  ##############################################################
  # カメラの設定
  conf = rs.config()
  # RGB
  conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  # 距離
  conf.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,30)
  # stream開始
  pipe = rs.pipeline()
  profile = pipe.start(conf)
  # Alignオブジェクト生成
  align_to = rs.stream.color
  align = rs.align(align_to)
  #############################################################
  return pipe, align

def IMAGE_End(DirNum):
  # 保存したいディレクトリ番号を指定して配列に集めた画像群をまとめて保存
  image = Image_Save(ImageBox, DirNum)
  cap.release()
  cv2.destroyAllWindows()

# sample test main code
#def main():
#    pipe, align = IMAGE_Init()
#    image = IMAGE_Get(pipe, align)
#    robot1, robot2, robot3 = IMAGE_GetPosi_GetAng(image)

# [result of main code]
#----------------------------------------------
#  Rotation Angle: 89 degrees
#  Rotation Angle: 84 degrees
#  Rotation Angle: 88 degrees
#[343, 175, 89] [421, 270, 84] [271, 256, 88]
#----------------------------------------------

if __name__ == "__main__":
    main()


