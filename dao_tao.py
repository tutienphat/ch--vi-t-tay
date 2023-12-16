from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
# Bước 1: Nạp dữ liệu từ bộ dữ liệu MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.summary()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Bước 4: Biên dịch mô hình
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Bước 5: Huấn luyện mô hình
model.fit(train_images, train_labels, epochs=1, batch_size=64)

# Bước 6: Đánh giá mô hình trên tập kiểm thử
test_loss, test_acc = model.evaluate(test_images, test_labels)

# In ra độ chính xác trên tập kiểm thử
print(test_acc)

# Bước 7: Lưu mô hình đã huấn luyện
model.save('mnist2.h5')
