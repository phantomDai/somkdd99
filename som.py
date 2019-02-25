import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import bone, pcolor, colorbar, plot, show
import os

def som_train():
    # importing the dataset
    print("导入数据...")
    dataset = pd.read_csv('data_train.csv')
    # get the matrix of features
    X = dataset.iloc[:, :-1].values
    # get the label of each sample
    Y = dataset.iloc[:, -1].values
    # get the labels
    labels = list(set(Y))

    print(labels)
    print(type(labels[0]))
    print(len(X), len(labels))
    # feature scaling
    print("归一化数据...")

    sc = MinMaxScaler(feature_range=(0, 1))
    X = sc.fit_transform(X)

    # training the SOM
    print("训练数据...")
    from minisom import MiniSom
    # this can be changed:
    som = MiniSom(x=7, y=7, input_len=len(X[0]), sigma=3.0, learning_rate=0.5,
                  neighborhood_function='triangle', random_seed=10)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=4000)

    # visualizing the result
    print("画图并记录日志")
    if os.path.exists('logs'):
        os.remove('logs')
    else:
        filelogs = open('logs', 'w', encoding="utf-8")

    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o', 'D', 'h', 'H', '_', '8', 'p', ',',
               '+', '.', 's', '*', 'd', '3', '0', '1', '2', 'v', '<', '7', '4', '5', '6']
    colors = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF', '#F5F5DC', '#FFE4C4',
              '#000000', '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0',
              '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C', '#00FFFF',
              '#00008B', '#008B8B']

    for i, x in enumerate(X):
        w = som.winner(x)
        filelogs.write("第" + str(i) + "数据对应的优胜神经元为：(" +
                   str(w[0]) + "," + str(1) + "),攻击类型的数据标识为： " +
                   str(Y[i]) + "\n")
        plot(w[0] + 0.5,
             w[1] + 0.5,
             markers[labels.index(Y[i])],
             markeredgecolor=colors[labels.index(Y[i])],
             markerfacecolor='None',
             markersize=10,
             markeredgewidth=3)
    filelogs.close()
    print("日志录入完毕请查看logs文件")
    show()

    print("测试模型...")
    test_cases = pd.read_csv('testcases.csv')
    print(test_cases)
    test_cases_X = test_cases.iloc[:, :-1].values
    print(len(test_cases_X))
    test_cases_Y = test_cases.iloc[:, -1].values
    print(len(test_cases_Y))
    test_cases_X = sc.fit_transform(test_cases_X)
    test_result_file = open('testresult', 'w', encoding="utf-8")
    for i, x in enumerate(test_cases_X):
        w = som.winner(x)
        test_result_file.write("第" + str(i) + "数据对应的优胜神经元为：(" +
                   str(w[0]) + "," + str(1) + "),正确的攻击类型的数据标识为： " +
                   str(test_cases_Y[i]) + "\n")

    print("模型测试完毕，请查看testlog文件")


if __name__ == '__main__':
    som_train()