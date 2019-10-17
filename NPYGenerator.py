# -*- coding=UTF-8 -*-
"""
@Time : 2019/5/28
@Author : gzf
@File : NPYGenerator.py
@brief: 处理原始轨迹数据，存储为npy格式
"""

import numpy as np
import os
import json
import sys
from absl import flags
from absl import app
import tensorflow as tf
import math


# def load_json(jsonfile):
#     """
#     读取单个json文件  load_json("data.json")
#     :param jsonfile: 文件名
#     :return: 轨迹位置点的list
#     """
#     with open(jsonfile, 'r') as load_f:
#         jdata = json.load(load_f)
#
#         traj_list = []
#         for i in range(np.size(jdata['FrameData'])):
#             position = list(jdata['FrameData'][i]['Positions'][2].values())
#             rotation = list(jdata['FrameData'][i]['Rotations'][2].values())
#             traj_list.append(list(np.concatenate((position, rotation), axis=0)))
#
#     return traj_list


def load_json(jsonfile, rota):
    """
    读取单个json文件  load_json("data.json")
    :param jsonfile: 文件名
    :param rota:    是否存储旋转四元数
    :return: 轨迹位置点的list
    """
    with open(jsonfile, 'r') as load_f:
        jdata = json.load(load_f)

        traj_list = []

        if rota:
            for i in range(np.size(jdata['Trajectory'])):
                position = list(jdata['Trajectory'][i]['Position'].values())
                rotation = list(jdata['Trajectory'][i]['Rotation'].values())
                traj_list.append(list(np.concatenate((position, rotation), axis=0)))
        else:
            for i in range(np.size(jdata['Trajectory'])):
                position = list(jdata['Trajectory'][i]['Position'].values())
                traj_list.append(list(position))

    return traj_list


def load_txt(traj_start):
    """
    读取TXT，用于采样时
    :param traj_start: 文件名
    :return: 轨迹起点和终点姿态
    """
    traj = np.loadtxt(traj_start)
    return traj


# json 转npy
def json2npy(json_files):
    trajs_path = os.listdir(json_files)

    for index in range(len(trajs_path)):
        json_data = np.array(load_json(json_files + '/' + trajs_path[index], True))
        path = './raw_data/npy/'+trajs_path[index][0:-4]+'npy'
        np.save(path, json_data)


# 计算两点间的距离
def calculate_distance(onePoint, otherPoint):
    d_x = otherPoint[0] - onePoint[0]
    d_y = otherPoint[1] - onePoint[1]
    d_z = otherPoint[2] - onePoint[2]
    # 计算两点之间的距离
    distance = math.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2)
    return distance


# 处理数据
def process_json(json_files, output_folder):
    """
    处理数据，存储有效轨迹
    :param json_files: 轨迹json文件夹
    :param output_folder: 输出结果文件夹
    :return:
    """
    threshold = 0.001

    trajs_path = os.listdir(json_files)
    for index in range(len(trajs_path)):
        print(index)
        print(json_files + '/' + trajs_path[index])
        with open(json_files + '/' + trajs_path[index], 'r') as load_f:
            jdata = json.load(load_f)

            # 创建字典，存储处理后的数据
            test_dict = {"Model": {}, "Trajectory": []}
            test_dict["Model"] = jdata["Model"]  # 模型数据不需要改变

            trajs_dict = []  # 存储轨迹

            short_num = 0
            long_num = 0
            record = 0
            for i in range(np.size(jdata['Trajectory']) - 1):
                p1 = list(jdata['Trajectory'][i][0]['Position'].values())
                p2 = list(jdata['Trajectory'][i + 1][0]['Position'].values())
                dis = calculate_distance(p1, p2)
                if dis > threshold:
                    record += 1
                else:
                    record = 0
                    short_num += 1

                if record > 10:
                    long_num += 1
                    traj_dict = {}  # 存储每个轨迹点
                    traj_dict["Position"] = jdata['Trajectory'][i][0]['Position']
                    traj_dict["Rotation"] = jdata['Trajectory'][i][0]['Rotation']
                    trajs_dict.append(traj_dict)

            test_dict['Trajectory'] = trajs_dict
            with open(output_folder + trajs_path[index], "w") as f:
                json.dump(test_dict, f)


# 处理单个轨迹
def data_generator(traj_json, pose_num, gap, or_rota):
    """
    处理单个轨迹
    例子：
    原始轨迹：A B C D E F G H I J K L M N
    采样轨迹（长度pose_num: 6）：① input_x:A B C D E F; input_y:B C D E F G; target:G
                               ② input_x:B C D E F G; input_y:C D E F G H; target:H
    :param traj_json: 文件名
    :param pose_num: 一个训练实例轨迹的位姿数
    :param gap: 采样轨迹起始点间隔，比如gap=2，轨迹1起始A点，轨迹2起始C点
    :param or_rota: 是否存储旋转四元数
    :return: 若干训练实例
    """
    json_data = load_json(traj_json, or_rota)    # 读取
    trajs = []  # 返回结果
    flag = False    # 结果是否有效

    if np.shape(json_data)[0] > pose_num:
        flag = True
        for i in range(1):  # int((np.shape(json_data)[0]-pose_num)/gap)
            traj_s = []
            traj_t = []
            targets = []

            for j in range(pose_num):
                traj_s.append(json_data[i*gap+j])
                traj_t.append(json_data[i*gap+j+1])
                targets.append(json_data[i*gap+pose_num])

            trajs.append([traj_s, traj_t, targets])
            # trajs.append([traj_s, traj_t])

    return flag, trajs  # trajs维数 (len(traj_json)-pose_num+1, 2, pose_num, 7)


# 生成数据集
def dataSet_generator(trajs_json, traj_start, save_trajs, pose_num, mode, or_rota):
    """
    处理文件夹下所有轨迹
    :param trajs_json: training模式轨迹数据路径
    :param traj_start: sample模式轨迹起始点
    :param save_trajs: npy存储路径
    :param pose_num: 每个训练实例的位姿数
    :param mode: training or sample
    :return:
    """
    trajs = []

    if mode == 'training':
        trajs_path = os.listdir(trajs_json)

        for index in range(len(trajs_path)):
            flag, traj = data_generator(trajs_json+'/'+trajs_path[index], pose_num, gap=2, or_rota=or_rota)
            if flag:
                trajs += traj

    elif mode == 'sample':
        traj = load_txt(traj_start)
        trajs += [traj]
        if or_rota:
            trajs = np.reshape(trajs, (1, 2, 1, 7))
        else:
            trajs = np.reshape(trajs, (1, 2, 1, 3))
    np.save(save_trajs, trajs)


FLAGS = flags.FLAGS

flags.DEFINE_string('trajs', "./raw_data/json_after/train", 'the path of trajs')
flags.DEFINE_string('traj_start', "./TrajNet_02/test.txt", 'the start pose for sample')
flags.DEFINE_string('trajs_save', "./TrajNet_02/traj_position.npy", 'the save path of trajs npy')
flags.DEFINE_string('mode', 'training', ' training or sample')
flags.DEFINE_integer('pose_num', 50, '轨迹中姿态数量')

def main(argv):
    # process_json('./raw_data/json', "./raw_data/json_after/")

    dataSet_generator(FLAGS.trajs, FLAGS.traj_start, FLAGS.trajs_save, FLAGS.pose_num, FLAGS.mode, False)
    if FLAGS.mode == "sample":
        print(np.load(FLAGS.trajs_save))
    else:
        print(np.shape(np.load(FLAGS.trajs_save)))

    # json2npy('./raw_data/json_after')


def DataForTest():
    test = load_json("raw_data/json_after/test/D-10212-2019-5-2415-17-15.json", False)
    np.savetxt("TrajNet_01/original.txt", test, fmt='%f', delimiter=' ')

    start_p = [test[0], test[-1]]
    np.savetxt("TrajNet_01/test.txt", start_p, fmt='%f', delimiter=' ')


if __name__ == '__main__':
    # DataForTest()
    app.run(main)

