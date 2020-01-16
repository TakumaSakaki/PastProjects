
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, y_train, x_test = load_fashionmnist()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1000)

def np_log(x):
    return np.log(np.clip(x, 1e-10, x))

def relu(x):
    return np.maximum(x, 0)

def deriv_relu(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    x -= x.max(axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def deriv_softmax(x):
    return softmax(x) * (1 - softmax(x))


class Dense:
    
    def __init__(self, in_dim, out_dim, function, deriv_function):
        self.W = np.random.uniform(low=-0.08, high=0.08,size=(in_dim, out_dim)).astype('float64')
        self.b = np.zeros(out_dim).astype('float64')
        self.function = function
        self.deriv_function = deriv_function

        self.x = None
        self.u = None

        self.dW = None
        self.db = None

        self.params_idxs = np.cumsum([self.W.size, self.b.size])

    def __call__(self, x):
        self.x = x
        self.u = np.matmul(self.x, self.W) + self.b
        return self.function(self.u)

    def b_prop(self, delta, W):
        self.delta = self.deriv_function(self.u) * np.matmul(delta, W.T)
        return self.delta
    
    def compute_grad(self):
        batch_size = self.delta.shape[0]
        self.dW = np.matmul(self.x.T, self.delta) / batch_size
        self.db = np.matmul(np.ones(batch_size), self.delta) / batch_size

    def get_params(self):
        return np.concatenate([self.W.ravel(), self.b], axis=0)
    
    def set_params(self, params):
        _W, _b = np.split(params, self.params_idxs)[:-1]
        self.W = _W.reshape(self.W.shape)
        self.b = _b
    
    def get_grads(self):
        return np.concatenate([self.dW.ravel(), self.db], axis=0)

    
def f_props(layers, x):
    for layer in layers:
        x = layer(x)
    return x

def b_props(layers, delta):
    batch_size = delta.shape[0]
    
    for i, layer in enumerate(layers[::-1]):
        if i == 0: # 出力層の場合
            layer.delta = delta # y - t
            layer.compute_grad() # 勾配の計算
        else: # 出力層以外の場合
            delta = layer.b_prop(delta, W) # 逆伝播
            layer.compute_grad() # 勾配の計算

        W = layer.W

def update_parameters(layers, eps):
    for layer in layers:
        layer.W -= eps * layer.dW
        layer.b -= eps * layer.db

layers = [
    Dense(784, 100, relu, deriv_relu),
    Dense(100, 100, relu, deriv_relu),
    Dense(100, 10, softmax, deriv_softmax)
]

def train_mnist(x, t, eps):
    # 順伝播
    y = f_props(layers, x)
    
    # 誤差の計算
    cost = (- t * np_log(y)).sum(axis=1).mean()
    
    # 逆伝播
    delta = y - t
    b_props(layers, delta)

    # パラメータの更新
    update_parameters(layers, eps)

    return cost

def valid_mnist(x, t):
    # 順伝播
    y = f_props(layers, x)
    
    # 誤差の計算
    cost = (- t * np_log(y)).sum(axis=1).mean()
    
    return cost, y

#学習
batch_size = 10
for epoch in range(10):
    x_train, y_train = shuffle(x_train, y_train)
    for i in range(int(len(x_train) / batch_size)):
        x = x_train[i * batch_size : (i+1) * batch_size]
        y = y_train[i * batch_size : (i+1) * batch_size]
        cost = train_mnist(x, y, eps = 0.01)
    
    cost, y_pred = valid_mnist(x_valid, y_valid)
    accuracy = accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(epoch + 1, cost, accuracy))

#テストデータの分類
y_pred = f_props(layers, x_test)
y_pred = np.argmax(y_pred, axis=1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('submission_pred.csv', header=True, index_label='id')