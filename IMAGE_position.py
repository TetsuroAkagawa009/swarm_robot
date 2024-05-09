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
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(-1, cv2.CAP_V4L)
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

def IMAGE_GetPosi(InputImage):
  kernel = np.ones((2, 2),np.uint8)
  robot1_posi = np.ones((2),np.uint8)
  robot2_posi = np.ones((2),np.uint8)

  # HSV色空間に変換
  hsv = cv2.cvtColor(InputImage, cv2.COLOR_BGR2HSV)
  # robot1のHSVの値域1
  hsv_min_1type = np.array([10,50,100]) 
  hsv_max_1type = np.array([20,170,230])
  mask1 = cv2.inRange(hsv, hsv_min_1type, hsv_max_1type)
  # robot2のHSVの値域2
  hsv_min_2type = np.array([0,150,100])
  hsv_max_2type = np.array([25,240,255])
  mask2 = cv2.inRange(hsv, hsv_min_2type, hsv_max_2type)
  # モフォロジー変換
  opening1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
  opening2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
  mu1 = cv2.moments(opening1, False)
  mu2 = cv2.moments(opening2, False)
  #重心から座標取得
  # robot1を見つけたとき
  if (mu1["m00"] != 0) and (mu2["m00"] == 0):
    robot1_posi = [int(mu1["m10"]/mu1["m00"]) , int(mu1["m01"]/mu1["m00"])]
  # robot2を見つけたとき
  if (mu1["m00"] == 0) and (mu2["m00"] != 0):
    robot2_posi = [int(mu2["m10"]/mu2["m00"]) , int(mu2["m01"]/mu2["m00"])]
  return robot1_posi, robot2_posi

def IMAGE_Get():
  # 画像データ取得
  ret, frame = cap.read()
  # numpy配列に変更
  frame_np =  np.array(frame, dtype="float32")
  print(frame_np.shape)
  # トリミング済み画像を配列に格納
  ImageBox.append(Image_Cut(frame))

def IMAGE_Init():
  cap = cv2.VideoCapture(-1, cv2.CAP_V4L)
  stepCount = 0
  ImageBox = []

def IMAGE_End(DirNum):
  # 保存したいディレクトリ番号を指定して配列に集めた画像群をまとめて保存
  Image_Save(ImageBox, DirNum)
  cap.release()
  cv2.destroyAllWindows()


