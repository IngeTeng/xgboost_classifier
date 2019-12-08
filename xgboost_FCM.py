import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

filename = 'Wafer'
#filename = 'ShapeletSim'
#filename = 'Meat'
TrainFcmMat = np.load(file='./data/'+filename+'TrainGCN.npy')
TestFcmMat = np.load(file='./data/'+filename+'TestGCN.npy')

if __name__=='__main__':
    x_train = []
    y_train = []

    x_test = []
    y_test = []
    #把矩阵中的每一个元素当做一个特征，将每个二维矩阵变成一维

    #训练数据
    for i in range(TrainFcmMat.shape[0]):
        #print(TrainFcmMat[i, 0, 0])
        if (TrainFcmMat[i, 0 , 0] == 1):
            y_train.append(1)
        else:
            y_train.append(-1)

        xtmp = []
        for j in range(TrainFcmMat.shape[1]):
            if j > 0:
                for k in range(TrainFcmMat.shape[2]):
                    xtmp.append(TrainFcmMat[i,j,k])
        x_train.append(xtmp)


    for i in range(TestFcmMat.shape[0]):

        if (TestFcmMat[i, 0 , 0] == 1):
            y_test.append(1)
        else:
            y_test.append(-1)

        xtmp = []
        for j in range(TestFcmMat.shape[1]):
            if j > 0:
                for k in range(TestFcmMat.shape[2]):
                    xtmp.append(TestFcmMat[i,j,k])
        x_test.append(xtmp)
    #pca降维
    model_pca = PCA(n_components=13)
    X_pca_train = model_pca.fit(x_train).transform(x_train)
    X_pca_test = model_pca.fit(x_test).transform(x_test)

    x_train = np.array(X_pca_train)
    y_train = np.array(y_train)

    x_test = np.array(X_pca_test)
    y_test = np.array(y_test)

    model = XGBClassifier(
        ilent=1,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        #nthread=4,# cpu 线程数 默认最大
        learning_rate= 0.3, # 如同学习率
        min_child_weight=1,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=6, # 构建树的深度，越大越容易过拟合
        gamma=0.4,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=0.8, # 随机采样训练样本 训练实例的子采样比
        max_delta_step = 0,#最大增量步长，我们允许每个树的权重估计。
        colsample_bytree=0.8, # 生成树时进行的列采样
        reg_lambda=0,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        reg_alpha=0, # L1 正则项参数
        #scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        objective= 'binary:logistic', #多分类的问题 指定学习任务和相应的学习目标
        #num_class=2, # 类别数，多分类与 multisoftmax 并用
        nthread=4,
        n_estimators=1000, #树的个数
        seed=1000 #随机种子
    )
    #eval_set = [(x_train, y_train)]
    eval_set = [(x_test, y_test)]
    model.fit(x_train, y_train, early_stopping_rounds=1000, eval_metric="auc", eval_set=eval_set, verbose=True)

    #获取验证集合结果
    evals_result = model.evals_result()
    y_test, y_pred = y_test, model.predict(x_test)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))

    #网格搜索
    param_test1 = {
        'max_depth':range(3,10,2),
        'gamma': [i / 10.0 for i in range(0, 5)],
        #'subsample': [i / 10.0 for i in range(6, 10)],
        #'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.3,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0.4,
        subsample=0.8,
        colsample_bytree=0.8,
        max_delta_step=0,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        reg_alpha=0,
        seed=1000
    ),
        param_grid = param_test1,
        scoring='roc_auc',
        n_jobs=4,
        iid=False,
        cv=5,
    )
    # print(y_train)
    # pass
    gsearch1.fit(x_train,y_train)

    # gsearch1.cv_results_ , gsearch1.best_params_,gsearch1.best_score_
    print("The best parameters are %s with a score of %0.4f"
          % (gsearch1.best_params_, gsearch1.best_score_))
    y_test, y_pred1 = y_test, gsearch1.predict(x_test)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred1))
