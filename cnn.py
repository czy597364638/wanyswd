import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    x_train = []
    y_train = []
    for i in range(1, 6):
        data_dict = unpickle(f'{data_dir}/data_batch_{i}')
        x_train.append(data_dict[b'data'])
        y_train.extend(data_dict[b'labels'])  
    
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)  
    
    test_dict = unpickle(f'{data_dir}/test_batch')
    x_test = test_dict[b'data']
    y_test = np.array(test_dict[b'labels'])  
    
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    
    return (x_train, y_train), (x_test, y_test)

data_dir = 'cifar-10-batches-py'

(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 构建CNN模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")