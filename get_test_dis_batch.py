# encoding=utf-8
import tensorflow as tf
import numpy as np
import os
import random
import math
import sys


HIP_CENTER = 7
SPINE = 4
SHOULDER_CENTER = 3;
HEAD = 20
SHOULDER_LEFT = 2
ELBOW_LEFT = 9
WRIST_LEFT = 11
HAND_LEFT = 13
SHOULDER_RIGHT = 1
ELBOW_RIGHT = 8
WRIST_RIGHT = 10
HAND_RIGHT = 12;
HIP_LEFT = 6
KNEE_LEFT = 15
ANKLE_LEFT = 17
FOOT_LEFT = 19
HIP_RIGHT = 5
KNEE_RIGHT = 14
ANKLE_RIGHT = 16
FOOT_RIGHT = 18

nui_skeleton_conn = {
    1: [HIP_CENTER, SPINE],
    2: [SPINE, SHOULDER_CENTER],
    3: [SHOULDER_CENTER, HEAD],
    # left arm 
    4: [SHOULDER_CENTER, SHOULDER_LEFT],
    5: [SHOULDER_LEFT, ELBOW_LEFT],
    6: [ELBOW_LEFT, WRIST_LEFT],
    7: [WRIST_LEFT, HAND_LEFT],
    # right arm
    8: [SHOULDER_CENTER, SHOULDER_RIGHT],
    9: [SHOULDER_RIGHT, ELBOW_RIGHT],
    10: [ELBOW_RIGHT, WRIST_RIGHT],
    11: [WRIST_RIGHT, HAND_RIGHT],
    # left leg
    12: [HIP_CENTER, HIP_LEFT],
    13: [HIP_LEFT, KNEE_LEFT],
    14: [KNEE_LEFT, ANKLE_LEFT],
    # right leg
    15: [HIP_CENTER, HIP_RIGHT],
    16: [HIP_RIGHT, KNEE_RIGHT],
    17: [KNEE_RIGHT, ANKLE_RIGHT]
}


skeleton_size = 17

class Batch:
    def __new__(cls, *args, **kwargs):
        return super(Batch, cls).__new__(cls, *args, **kwargs)
    pass

    def __init__(self):
        self.a = np.zeros(np.array([39 * 67 * 67]))
        self.b = np.zeros(np.array([140]))
    pass
pass

class Data:
    def __new__(cls, *args, **kwargs):
        return super(Data, cls).__new__(cls, *args, **kwargs)
    pass

    def __init__(self, dir):
        self.dir = dir
        self.GBatch=[]
    pass

    def getUni(self, j, k):
        uniNum = set()
        uniNum.add(nui_skeleton_conn[j][0] - 1)
        uniNum.add(nui_skeleton_conn[j][1] - 1)
        uniNum.add(nui_skeleton_conn[k][0] - 1)
        uniNum.add(nui_skeleton_conn[k][1] - 1)
        length = len(uniNum)
        num1 = uniNum.pop()
        num2 = uniNum.pop()
        num3 = uniNum.pop()
        num4 = 0
        if length == 3:
            return 3, num1, num2, num3, num4
        if length == 4:
            num4 = uniNum.pop()
            return 4, num1, num2, num3, num4
    pass

    def getText(self, fileName):
        self.route = self.dir + fileName
        floatNum = np.loadtxt(self.route)
        size_3 = floatNum.size / 80  # number of frame
        new_feature = np.zeros(np.array([size_3, 20, 67]))  # not sure about the second dimension
        floatNum = floatNum.reshape(size_3, 20, 4)  # frame,joints,dimension
        floatNum = floatNum[:, :, 0:3]  # position(x,y,z)
        coords = floatNum.reshape(size_3, 20, 3)
        for i in range(size_3):  # get all frame
            # feature1:get angle feature
            countOfNum = np.zeros(np.array([20]))
            tmp_joint = floatNum[i, :, :]
            for j in range(1, skeleton_size + 1):
                for k in range(j + 1, skeleton_size + 1):
                    tmp_vec1 = tmp_joint[nui_skeleton_conn[j][0] - 1, :] - tmp_joint[nui_skeleton_conn[j][1] - 1, :]
                    tmp_vec2 = tmp_joint[nui_skeleton_conn[k][0] - 1, :] - tmp_joint[nui_skeleton_conn[k][1] - 1, :]
                    if (np.linalg.norm(tmp_vec1) * np.linalg.norm(tmp_vec2)) != 0:
                        tmp_angle = np.dot(tmp_vec1, tmp_vec2) / (np.linalg.norm(tmp_vec1) * np.linalg.norm(tmp_vec2))
                    else:
                        print fileName, i, tmp_vec1, tmp_vec2, i, j, k
                        tmp_angle = 0
                    num, num1, num2, num3, num4 = self.getUni(j, k)
                    new_feature[i, num1, int(countOfNum[num1])] = tmp_angle
                    new_feature[i, num2, int(countOfNum[num2])] = tmp_angle
                    new_feature[i, num3, int(countOfNum[num3])] = tmp_angle
                    countOfNum[num1] = countOfNum[num1] + 1
                    countOfNum[num2] = countOfNum[num2] + 1
                    countOfNum[num3] = countOfNum[num3] + 1
                    if num == 4:
                        new_feature[i, num4, int(countOfNum[num4])] = tmp_angle
                        countOfNum[num4] = countOfNum[num4] + 1
            maxNum = int(np.max(countOfNum, 0))

        def dis(a, b, t):
            ans = np.sqrt((coords[t][a][0]-coords[t][b][0])*(coords[t][a][0]-coords[t][b][0])+
            (coords[t][a][1] - coords[t][b][1]) * (coords[t][a][1] - coords[t][b][1])+
            (coords[t][a][2] - coords[t][b][2]) * (coords[t][a][2] - coords[t][b][2]))
            return ans
        pass
        #feature2 relative distance
        for i in range(size_3):
            for j in range(20):
                new_feature[i][j][maxNum] = dis(j,HEAD-1,i)
                new_feature[i][j][maxNum+1] = dis(j, SHOULDER_CENTER-1, i)
                new_feature[i][j][maxNum+2] = dis(j, SHOULDER_LEFT-1, i)
                new_feature[i][j][maxNum+3] = dis(j, SHOULDER_RIGHT-1, i)
                new_feature[i][j][maxNum+4] = dis(j, SPINE-1, i)
                new_feature[i][j][maxNum+5] = dis(j, HIP_CENTER-1, i)
                new_feature[i][j][maxNum+6] = dis(j, HIP_LEFT-1, i)
                new_feature[i][j][maxNum+7] = dis(j, HIP_RIGHT-1, i)

        def offset(a, t1, t2):
            ans = np.sqrt((coords[t1][a][0]-coords[t2][a][0])*(coords[t1][a][0]-coords[t2][a][0])+
            (coords[t1][a][1] - coords[t2][a][1]) * (coords[t1][a][1] - coords[t2][a][1])+
            (coords[t1][a][2] - coords[t2][a][2]) * (coords[t1][a][2] - coords[t2][a][2]))
            return ans
        pass
        # feature3 offset distance
        for i in range(1,size_3):
            for j in range(20):
                new_feature[i][j][maxNum+8] = offset(j, i, i-1)


        ans = np.zeros(np.array([20, 67, size_3 - 1]))
        for t1 in range(size_3 - 1):
            for t2 in range(20):
                for t3 in range(67):
                    ans[t2][t3][t1] = new_feature[t1][t2][t3]

        return ans
    pass

    def getCovInput(self, fileName):
        feature = self.getText(fileName)  # [joint,feature,frame]
        convInput = np.zeros(np.array([39, 67, feature.shape[2]]))
        convInput[0, :, :] = feature[SPINE - 1, :, :]
        convInput[1, :, :] = feature[SHOULDER_CENTER - 1, :, :]
        convInput[2, :, :] = feature[HEAD - 1, :, :]
        convInput[3, :, :] = feature[SHOULDER_CENTER - 1, :, :]
        convInput[4, :, :] = feature[SHOULDER_RIGHT - 1, :, :]
        convInput[5, :, :] = feature[ELBOW_RIGHT - 1, :, :]
        convInput[6, :, :] = feature[WRIST_RIGHT - 1, :, :]
        convInput[7, :, :] = feature[HAND_RIGHT - 1, :, :]
        convInput[8, :, :] = feature[WRIST_RIGHT - 1, :, :]
        convInput[9, :, :] = feature[ELBOW_RIGHT - 1, :, :]
        convInput[10, :, :] = feature[SHOULDER_RIGHT - 1, :, :]
        convInput[11, :, :] = feature[SHOULDER_CENTER - 1, :, :]
        convInput[12, :, :] = feature[SHOULDER_LEFT - 1, :, :]
        convInput[13, :, :] = feature[ELBOW_LEFT - 1, :, :]
        convInput[14, :, :] = feature[WRIST_LEFT - 1, :, :]
        convInput[15, :, :] = feature[HAND_LEFT - 1, :, :]
        convInput[16, :, :] = feature[WRIST_LEFT - 1, :, :]
        convInput[17, :, :] = feature[ELBOW_LEFT - 1, :, :]
        convInput[18, :, :] = feature[SHOULDER_LEFT - 1, :, :]
        convInput[19, :, :] = feature[SHOULDER_CENTER - 1, :, :]
        convInput[20, :, :] = feature[SPINE - 1, :, :]
        convInput[21, :, :] = feature[HIP_CENTER - 1, :, :]
        convInput[22, :, :] = feature[HIP_RIGHT - 1, :, :]
        convInput[23, :, :] = feature[KNEE_RIGHT - 1, :, :]
        convInput[24, :, :] = feature[ANKLE_RIGHT - 1, :, :]
        convInput[25, :, :] = feature[FOOT_RIGHT - 1, :, :]
        convInput[26, :, :] = feature[ANKLE_RIGHT - 1, :, :]
        convInput[27, :, :] = feature[KNEE_RIGHT - 1, :, :]
        convInput[28, :, :] = feature[HIP_RIGHT - 1, :, :]
        convInput[29, :, :] = feature[HIP_CENTER - 1, :, :]
        convInput[30, :, :] = feature[HIP_LEFT - 1, :, :]
        convInput[31, :, :] = feature[KNEE_LEFT - 1, :, :]
        convInput[32, :, :] = feature[ANKLE_LEFT - 1, :, :]
        convInput[33, :, :] = feature[FOOT_LEFT - 1, :, :]
        convInput[34, :, :] = feature[ANKLE_LEFT - 1, :, :]
        convInput[35, :, :] = feature[KNEE_LEFT - 1, :, :]
        convInput[36, :, :] = feature[HIP_LEFT - 1, :, :]
        convInput[37, :, :] = feature[HIP_CENTER - 1, :, :]
        convInput[38, :, :] = feature[SPINE - 1, :, :]
        return convInput
    pass

    def getFile(self):
        for i in range(1,150): 
            for j in range(1,50):
                name = 'a' + str(i) + '_r' + str(j)
                if os.path.exists(self.dir + name) == False:
                    continue
                batch = Batch()
                ConvInput = self.getCovInput(name)
                ConvInput = ConvInput.reshape([39 * 67 * 67])
                batch.a = ConvInput
                batch.b[i - 1] = 1
                self.GBatch.append(batch)
            print i
        return self.GBatch
    pass

    def getInputBatch(self):
        first = []
        second = []
        length=len(self.GBatch)
        for i in range(length):
            copy = self.GBatch[i].a.reshape([39 * 67 * 67])
            for j in range(39 * 67 * 67):
                first.append(copy[j])
        for i in range(length):
            copy = self.GBatch[i].b.reshape([140])
            for j in range(140):
                second.append(copy[j])
        return np.array(first), np.array(second)
    pass
pass






data = Data('./test_67/')
data.getFile()
input1,input2=data.getInputBatch()

if not os.path.exists('./test_dis_67_input/'):
    os.mkdir('./test_dis_67_input/')

input1=np.save('./test_dis_67_input/test1.npy',input1)
input2=np.save('./test_dis_67_input/test2.npy',input2)