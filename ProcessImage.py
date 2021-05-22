import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
class Processer:
    def __init__(self):
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [2, 3], [2, 3]]
        self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        self.model = get_testing_model()
        self.model.load_weights('model/keras/model.h5')
        self.filepath = ""
        self.input_image = ""
        self.people_count = 0
        self.subset = -1 * np.ones((0, 20))
        self.params, self.model_params = config_reader()
        self.candidate = None
    def cv_imread(self,input_image):
        self.filepath=input_image
        cv_img = cv2.imdecode(np.fromfile(self.filepath, dtype=np.uint8), -1)
        return cv_img
    def process (self):
        oriImg = self.cv_imread(self.input_image)  # B,G,R order
        multiplier = 0.8*self.model_params['boxsize']/oriImg.shape[0]

        scale = multiplier
        print(scale)
        ##图片x 缩放到294
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(imageToTest.shape)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, self.model_params['stride'],self.model_params['padValue'])
        cv2.imwrite("pad.png",imageToTest)

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
        output_blobs = self.model.predict(input_img)

        #提取输出，调整大小并删除填充
        ##heatmap[i][j][k] 是 (j,i)处为姿势k的置信度
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=self.model_params['stride'], fy=self.model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        ## paf[i][j][k]表示[j][i]处位于姿势k的概率
        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=self.model_params['stride'], fy=self.model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)


        all_peaks = []##每个关键点的可能性最大的坐标
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap[:, :, part]## 每一个姿势的置信矩阵
            map = gaussian_filter(map_ori, sigma=3)## 函数

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > self.params['thre1']))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(self.mapIdx)):
            score_mid = paf[:, :, [x - 19 for x in self.mapIdx[k]]]

            candA = all_peaks[self.limbSeq[k][0] - 1]##A点集
            candB = all_peaks[self.limbSeq[k][1] - 1]##B点集
            nA = len(candA)
            nB = len(candB)
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2]) #两点之间的坐标向量
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])## 向量的模

                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)## 单位向量

                        ##一个资态向量组，长度为10，
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                        vec_x = np.array(
                            [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                             for I in range(len(startend))])


                        vec_y = np.array(
                            [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                             for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        # print(vec[0],vec[1],score_midpts)

                        # score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        #      oriImg.shape[0] / norm - 1, 0)
                        ## A,B之间的平均权值
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts)

                        ## 80%以上的点的权值大于阈值
                        criterion1 = len(np.nonzero(score_midpts > self.params['thre2'])[0]) > 0.8 * len(
                            score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:## 构造候选图
                            connection_candidate.append([i, j, score_with_dist_prior,
                                                         score_with_dist_prior + candA[i][2] + candB[j][2]])
                ##按照权值从大到小排序
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]]) ##第A类点的第i个和第B类点的第j个之间的权值
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        self.candidate = candidate
        for k in range(len(self.mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(self.limbSeq[k]) - 1
                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                                  connection_all[k][i][2]
                        subset = np.vstack([subset, row])


        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.5:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        canvas = self.cv_imread(self.input_image)  # B,G,R order
        only_Pose = np.zeros((canvas.shape[0],canvas.shape[1],3), np.uint8)
        only_Pose.fill(0)
        maxx=len(all_peaks[0])
        for i in range(18):
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, all_peaks[i][j][0:2], 4, self.colors[i], thickness=-1)
                cv2.circle(only_Pose,all_peaks[i][j][0:2], 4, self.colors[i], thickness=-1)
                #cv2.putText(canvas,"("+str(all_peaks[i][j][0])+","+str(all_peaks[i][j][1])+")",all_peaks[i][j][0:2],cv2.FONT_HERSHEY_SIMPLEX,0.25,(255, 0, 0),1)
        ##骨架宽度
        stickwidth = 2
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(self.limbSeq[i]) - 1]  ##第n个人的第i个关键点的index
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                cur_pose = only_Pose.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                # print(n,i,Y[0],X[0],Y[1],X[1]) ##第n个人第i个肢体的向量

                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
                cv2.fillConvexPoly(cur_pose, polygon, self.colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
                only_Pose = cv2.addWeighted(only_Pose, 0.4, cur_pose,0.6,0)
                self.subset = subset
        ##给每个人分配id
        for n in range(len(subset)):
            index = subset[n][np.array(self.limbSeq[1]) - 1]
            if -1 in index:
                index = subset[n][np.array(self.limbSeq[2]) - 1]
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            cv2.putText(canvas,str(n+1),(int(Y[0]),int(X[0]-5)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(68,255, 51),5)
        person_count = len(subset)
        cv2.imwrite("only_Pose.png", only_Pose)
        cv2.imwrite("result.png", canvas)
        return canvas,person_count

    def Pose(self,Img_path):

        self.input_image = Img_path

        print('start processing...')

        res = self.process()
        canvas = res[0]
        self.people_count = res[1]
        canvas = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_CUBIC)
        org = self.cv_imread(self.input_image)
        org = cv2.resize(org, (256, 256), interpolation=cv2.INTER_CUBIC)
        print("success!")
        return canvas, self.people_count
