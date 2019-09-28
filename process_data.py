# -*- coding: utf-8 -*-
# @Time : 2019/7/4 16:30
# @Author : gzf
# @File : process_data.py
# @brif : 处理数据

import json
import numpy as np
import os
# 读取json
def load_json(jsonfile):
    """
    读取单个json文件  load_json("data.json")
    :param jsonfile: 文件名
    :return: 轨迹位置点的list
    """
    with open(jsonfile, 'r') as load_f:
        jdata = json.load(load_f)

        traj_list = []
        for i in range(np.size(jdata['Trajectory'])):
            position = list(jdata['Trajectory'][i][0]['Position'].values())
            rotation = list(jdata['Trajectory'][i][0]['Rotation'].values())
            traj_list.append(list(np.concatenate((position, rotation), axis=0)))

    return traj_list

def compute_distance(point1, point2):
    return np.sqrt(np.square(point1[0]-point2[0])+np.square(point1[1]-point2[1])+np.square(point1[2]-point2[2]))

def distance_p2p(jsonfile):
    min = 0.003
    low_min = 10
    res_list = []
    json_data = load_json(jsonfile)
    for i in range(np.shape(json_data)[0]-1):
        point1 = json_data[i][0:3]
        point2 = json_data[i+1][0:3]
        dis = compute_distance(point1, point2)
        if dis < min:
            low_min += 1
            if low_min > 10:
                low_min = 10
        else:
            low_min -= 1
            if low_min < 0:
                low_min = 0

        if(low_min < 10):
            res_list.append(json_data[i+1])
    f = open(jsonfile[0:-4]+'txt', 'w')
    for i in res_list:
        k = ' '.join([str(j) for j in i])
        f.write(k + "\n")
    f.close()



def process(trajs_json):
    trajs_path = os.listdir(trajs_json)

    for index in trajs_path:
        if os.path.splitext(index)[1] == ".json":
            distance_p2p(trajs_json+'/'+index)
            print("-----------")

distance_p2p("raw_data/json/D-10000-2019-5-2216-23-27.json")

# process("./raw_data/json")