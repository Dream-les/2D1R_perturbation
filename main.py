import User.utils.dataloader
import User.nn.TDOR as TDOR
import numpy as np
from User.nn.NonlinearNetwork import Network
from User.nn.optim import Function
import matplotlib.pyplot as plt
import User.utils.dataloader as dl
import gc
import pickle
from User.nn.TDOR import Behaviour
from matplotlib import animation


# def train_function and test_function
def train_function(nn_in, x_train, y_train, f_obj_in, lr_in=0.001, ):  # TODO: Decorator is more suitable?
    nn_in.forward(x_train, y_train, f_obj_in, lr=lr_in)
    nn_in.step()
    # return np.mean(nn_in.get_loss), count


def test_function(nn_in, x_in, y_in, f_obj_in, lr_in=0.001, ):
    y_pred, loss = nn_in.forward(x_in, y_in, f_obj_in, lr=lr_in)
    # return np.mean(nn_in.get_loss,axis=1), nn_in.get_count


def update(data_in, epoch_in, ax_in):
    xdata = np.arange(0, epoch_in)
    ydata = data_in
    ax_in.set_data(xdata, ydata)


def init_figure(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)


# Load the data
optd = dl.Datasets()
train, test = optd('optdigits', preprocess=True)

# Define the neural network and initialization
LOAD_MODEL = True  # True or False
LOAD_EVAL = True  # True or False
model_save_path = 'F:\\Desktop\\temp\\python\\2D1R_PP\\pythonProject\\model\\weights.pkl'


if LOAD_MODEL:
    # nn.load_model(model_save_path)
    with open(model_save_path, 'rb') as f:
        nn = pickle.load(f)
else:
    # layer_list = [TDOR.Layer2Layer(), TDOR.Behaviour(8 * 8, 20),
    #               TDOR.Layer2Layer(), TDOR.Behaviour(20, 10),
    #               TDOR.Layer2Layer(), ]
    layer_list = [TDOR.Layer2Layer(), TDOR.Behaviour(8 * 8, 10),
                  TDOR.Layer2Layer(), ]
    nn = Network(layer_list)

f_obj = Function("MSE", "sigmoid", )

# Train the neural network
loss_vector = np.array([])  # TODO: Check
count = 0  # count the number of correct predictions
learning_rate = 2e-6  # 1e-6, 2e-6, 1e-7, 8e-7
train_size = 3823  # 3823
epochs = 100
test_size = 1797

eval_path = 'F:\\Desktop\\temp\\python\\2D1R_PP\\pythonProject\\model\\eval.pkl'
if LOAD_EVAL:
    with open(eval_path, 'rb') as f:
        eval_dict = pickle.load(f)
    loss_value = eval_dict['loss_value'].tolist()
    accuracy = eval_dict['accuracy']
    test_accuracy = eval_dict['test_accuracy']
    test_loss_value = eval_dict['test_loss_value']
else:
    test_accuracy = []
    test_loss_value = []
    accuracy = []
    loss_value = []

# nn.limit_weights()

for epoch in range(epochs):
    print("################Epoch {}/{}################".format(epoch + 1, epochs))
    for i in range(train_size):
        num = np.random.randint(0, train_size)
        x = train['data'][num, :, :].reshape(-1, 8 * 8)
        y = train['target'][num, 0, :].reshape(-1, 10)
        train_function(nn, x, y, f_obj, learning_rate, )
        # if i % 1000 == 0:
        #     nn.limit_weights()

    # nn.limit_weights()
    accuracy.append(nn.get_count / train_size)  #  * (epoch + 1))
    loss_value.append(np.mean(np.mean(nn.get_loss, axis=1)[1:].reshape(-1, train_size), axis=1))
    nn.reset_count_loss()
    print("train Epoch: ", len(loss_value), " Loss: ", loss_value[-1], " Accuracy: ", accuracy[-1])
    gc.collect()
    # if epoch % 5 == 0:
    #     learning_rate *= 0.2

    # Test the neural network
    """
    TESTING THE NETWORK
    """
    test_count = 0
    for i in range(test_size):
        num = np.random.randint(0, test_size)
        x = test['data'][num, :, :].reshape(-1, 8 * 8)
        y = test['target'][num, 0, :].reshape(-1, 10)
        test_function(nn, x, y, f_obj, learning_rate, )
    test_accuracy.append(nn.get_count / test_size)
    test_loss_value.append(np.mean(np.mean(nn.get_loss, axis=1)[1:].reshape(-1, test_size), axis=1))
    nn.reset_count_loss()
    print("test Epoch: ", len(test_loss_value), " Loss: ", test_loss_value[-1], " Accuracy: ", test_accuracy[-1])

print("################Training Finished#")
loss_value = np.array(loss_value)

# Plot the loss value and accuracy
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# fig.set_size_inches(10, 10)
ax[0, 0].plot(loss_value, 'r-o')
ax[0, 0].set_title('Loss value')
ax[0, 1].plot(accuracy, 'b-o')
ax[0, 1].set_title('Train accuracy')
ax[1, 0].plot(test_accuracy, 'b-o')
ax[1, 0].set_title('Test accuracy' + str(nn.get_y_pred.argmax(axis=1)))
ax[1, 1].imshow(x.reshape(8, 8))
ax[1, 1].set_title('Input Image' + str(y.argmax(axis=1)))
plt.show()
# Animation
# ani = animation.FuncAnimation(
#     fig=fig,
#     func=update,
#     frames=np.linspace(0, 5, 100),  # [1, 2, 3]
#     init_func=init_figure,
#     interval=5,  # 每隔多少时间生成一帧图像，单位是ms
#     repeat=True,  # 设置不重复，但是还是重复较好
# )

# Save the evaluation results
with open(eval_path, 'wb') as f:
    pickle.dump({'loss_value': loss_value, 'accuracy': accuracy, 'test_accuracy': test_accuracy,
                 'test_loss_value': test_loss_value}, f)

# Save the model
nn.save_model(model_save_path)

print("Hello World!")

'''# Load the data
optd = dl.Datasets()
train, test = optd('optdigits', preprocess=True)


# Define the neural network
# layer_list = [TDOR.Layer2Layer(), TDOR.Linear(8*8, 20),
#               TDOR.Layer2Layer(), TDOR.Linear(20, 10), ]
layer_list = [TDOR.Layer2Layer(), TDOR.Behaviour(8*8, 20),
              TDOR.Layer2Layer(), TDOR.Behaviour(20, 10),
              TDOR.Layer2Layer(), ]
nn = Network(layer_list)
f_obj = Function("MSE", "sigmoid", )
x = train['data'][7, :, :].reshape(-1, 8*8)
y = train['target'][7, 0, :].reshape(-1, 10)
loss_value = []


# Train the neural network
for epoch in range(50):
    for i in range(100):
        nn.forward(x, y, f_obj, lr=0.001)
        nn.step()
        loss_value.append(np.mean(nn.get_loss))
    nn.limit_weights()
    gc.collect()
loss_value = np.array(loss_value)

fig, [ax1, ax2] = plt.subplots(1, 2)
ax1.plot(loss_value, 'r-o')
ax1.set_title('Loss value' + str(nn.get_y_pred.argmax(axis=1)))
ax2.imshow(x.reshape(8, 8))
ax2.set_title('Input Image' + str(y.argmax(axis=1)))
plt.show()'''
