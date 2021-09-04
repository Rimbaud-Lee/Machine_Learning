
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
import operator
import time


# 取消 PyCharm 对 DataFrame 的显示限制
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# 解析数据
def file_matrix(filepath):
    dt = pd.read_csv(filepath)
    date_mat = np.zeros((dt.shape[0], 3))
    # 分类标签向量
    class_label_vec = []
    col_val = list(dt)
    # 取前3列填充进0矩阵，即特征矩阵
    for i in col_val[:3]:
        ind = col_val.index(i)
        date_mat[:, ind] = dt[i]
    for row in dt.itertuples():
        att = getattr(row, 'Attitude')
        if att == "didntLike":
            class_label_vec.append(1)
        elif att == "smallDoses":
            class_label_vec.append(2)
        elif att == "largeDoses":
            class_label_vec.append(3)
    return date_mat, class_label_vec


# 数据可视化
def visualize(date_mat, class_label_vec):
    # 字体样式
    font = {"family" : "MicroSoft YaHei",
            "weight" : 6,
            "size" : 6}
    matplotlib.rc("font", **font)
    # 子图板式
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8), dpi=300)
    colors_lab = []
    for i in class_label_vec:
        if i == 1:
            colors_lab.append("black")
        elif i == 2:
            colors_lab.append("orange")
        elif i == 3:
            colors_lab.append("red")
    # 图一，以矩阵的第一(飞行里程)、第二(游戏)列数据画散点数据，散点大小为15，透明度为0.5
    axs[0][0].scatter(x=date_mat[:, 0], y=date_mat[:, 1], color=colors_lab, s=15, alpha=.5)
    axs0_title = axs[0][0].set_title("每年获得的飞行常客里程数与玩视频游戏所消耗时间占比")
    axs0_xlabel = axs[0][0].set_xlabel("每年获得的飞行常客里程数")
    axs0_ylabel = axs[0][0].set_ylabel("玩视频游戏所消耗时间占")
    plt.setp(axs0_title, size=8, weight="bold", color="black")
    plt.setp(axs0_xlabel, size=7, weight="bold", color="black")
    plt.setp(axs0_ylabel, size=7, weight="bold", color="black")
    # 图二，以矩阵的第一(飞行里程)、第三(冰激凌)列数据画散点数据
    axs[0][1].scatter(x=date_mat[:, 0], y=date_mat[:, 2], color=colors_lab, s=15, alpha=.5)
    axs1_title = axs[0][1].set_title("每年获得的飞行常客里程数与每周消费的冰激凌公升数")
    axs1_xlabel = axs[0][1].set_xlabel("每年获得的飞行常客里程数")
    axs1_ylabel = axs[0][1].set_ylabel("每周消费的冰激凌公升数")
    plt.setp(axs1_title, size=8, weight="bold", color="black")
    plt.setp(axs1_xlabel, size=7, weight="bold", color="black")
    plt.setp(axs1_ylabel, size=7, weight="bold", color="black")
    # 图三，以矩阵的第二(游戏)、第三(冰激凌)列数据画散点数据
    axs[1][0].scatter(x=date_mat[:, 1], y=date_mat[:, 2], color=colors_lab, s=15, alpha=.5)
    axs2_title = axs[1][0].set_title("每年获得的飞行常客里程数与每周消费的冰激凌公升数")
    axs2_xlabel = axs[1][0].set_xlabel("每年获得的飞行常客里程数")
    axs2_ylabel = axs[1][0].set_ylabel("每周消费的冰激凌公升数")
    plt.setp(axs2_title, size=8, weight="bold", color="black")
    plt.setp(axs2_xlabel, size=7, weight="bold", color="black")
    plt.setp(axs2_ylabel, size=7, weight="bold", color="black")
    # 设置图例样式
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    plt.show()


# 数值归一化
def auto_norm(date_mat):
    min_vals = date_mat.min(0)
    max_vals = date_mat.max(0)
    ranges = max_vals - min_vals
    norm_data = np.zeros(np.shape(date_mat))
    l = date_mat.shape[0]
    norm_data = date_mat - np.tile(min_vals, (l, 1))
    norm_data = norm_data / np.tile(ranges, (l, 1))
    return norm_data


# 分类器——k-NN算法
def classify(inX, dataset, labels, k):
    """
    :param inX: 用于测试的数据(测试集)
    :param dataset: 用于训练的数据(训练集)
    :param labels: 分类标签
    :param k: 距离最小的k个点
    :return:
    """
    # 计算距离
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(inX, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(1)
    distances = sq_distances**0.5
    sorted_distances = distances.argsort()
    # 计数字典
    class_count = {}
    for i in range(k):
        votelabel = labels[sorted_distances[i]]
        class_count[votelabel] = class_count.get(votelabel, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 次数最多的类别
    return sorted_class_count[0][0]


# 验证器
def test(filepath):
    date_mat, class_label_vec = file_matrix(filepath)
    ratio = 0.10
    norm_mat = auto_norm(date_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ratio)
    # 分类错误计数
    error_count = 0.0
    for i in range(num_test_vecs):
        # 前 num_test_vecs 个数据作为测试集,后 m-num_test_vecs 个数据作为训练集，取 k 为7
        classifier_result = classify(
            norm_mat[i, :],
            norm_mat[num_test_vecs:, :],
            class_label_vec[num_test_vecs:],
            7
        )
        print("分类结果:%d\t真实类别:%d" % (classifier_result, class_label_vec[i]))
        if classifier_result != class_label_vec[i]:
            error_count += 1.0
    # 输出正确率
    print("正确率:%f%%" % ((1.00 - error_count / float(num_test_vecs)) * 100))
    # visualize(date_mat, class_label_vec)


if __name__ == '__main__':
    filepath = r"C:\Users\ASUS\Desktop\datingKNN\datingTestSet.csv"
    test(filepath)
    print("\n")
    print("程序运行时间为：", time.process_time(), "秒")

