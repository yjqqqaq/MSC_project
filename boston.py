from sklearn.datasets import load_boston
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
dataset = load_boston()
#导入数据
x_data, y_data = dataset.data, dataset.target.reshape(-1, 1)

#分离测试集和训练集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, random_state = 0, test_size = 0.25)
scaler = preprocessing.MinMaxScaler()
#均衡数据，加速收敛
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)
model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
#输出对模型的评分
print(r2_score(y_test, y_predict))


