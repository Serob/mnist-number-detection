import keras
from keras import regularizers
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from train.loader import MnistDataloader

data_dir = '../data'
# training_images = join(data_dir, 'train-images.idx3-ubyte')
train_images_path = '../data/train-images.idx3-ubyte'
train_labels_path = '../data/train-labels.idx1-ubyte'
test_images_path = '../data/t10k-images.idx3-ubyte'
test_labels_path = '../data/t10k-labels.idx1-ubyte'


data_loader = MnistDataloader(
	train_images_path=train_images_path,
	train_labels_path=train_labels_path,
	test_images_path=test_images_path,
	test_labels_path=test_labels_path
)

(train_images, train_labels), (test_images, test_labels) = data_loader.load_all()

print(f'TensorFlow info: {device_lib.list_local_devices()}\n')

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(320, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(320, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
	keras.layers.Dense(64, activation='relu'),
	keras.layers.Dense(10)
])

print(model.summary())

model.compile(
	optimizer='adam',
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy']
)


for i in range(10):
	model.fit(
		train_images,
		train_labels,
		epochs=14,
		validation_data=(test_images, test_labels)
	)  # validation_data=(test_images, test_labels)

	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

	print('\nTest accuracy:', test_acc)

	accuracy_short = '{:.4f}'.format(test_acc)
	model.save(f'../models/mnist-{accuracy_short}.keras')


def plot_image(images, labels, index: int):
	plt.figure()
	plt.imshow(images[index])  # , cmap=plt.cm.binary
	plt.title(labels[index], fontsize=20)
	plt.colorbar()
	plt.show()


# for i in range(5):
# 	plot_image(test_images, test_labels, i)



