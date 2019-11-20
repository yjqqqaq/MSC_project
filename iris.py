from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

dataset = load_iris()
#导入数据
x_data, y_data = dataset.data, dataset.target.reshape(-1, 1)

print(x_data.shape)
print(y_data.shape)

#分离测试集和训练集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, random_state = 0, test_size = 0.25)
scaler = preprocessing.MinMaxScaler()
#均衡数据，加速收敛
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
model = KNeighborsClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

#输出对模型的评分
print(r2_score(y_test, y_predict))


