
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, y_train, x_test = load_fashionmnist()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1000)

def np_log(x):
    # WRITE ME

def relu(x):
    # WRITE ME

def deriv_relu(x):
    # WRITE ME

def softmax(x):
    # WRITE ME

def deriv_softmax(x):
    # WRITE ME

class Dense:
    # WRITE ME

def f_props(layers, x):
    # WRITE ME

def b_props(layers, delta):
    # WRITE ME

def update_parameters(layers, eps):
    # WRITE ME

layers = [
    # WRITE ME
]

def train_mnist(x, t, eps):
    # WRITE ME

def valid_mnist(x, t):
    # WRITE ME

for epoch in range(5):
    x_train, y_train = shuffle(x_train, y_train)
    # オンライン学習
    # WRITE ME

y_pred = np.argmax(y_pred, axis=1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/chap04/sample_submission_pred.csv', header=True, index_label='id')
