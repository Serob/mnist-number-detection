from train.loader import MnistDataloader
import matplotlib.pyplot as plt
from os.path import join

data_dir = '../data'
# training_images = join(data_dir, 'train-images.idx3-ubyte')
train_images = '../data/train-images.idx3-ubyte'
train_labels = '../data/train-labels.idx1-ubyte'
test_images = '../data/t10k-images.idx3-ubyte'
test_labels = '../data/t10k-labels.idx1-ubyte'


data_loader = MnistDataloader(
	train_images_path=train_images,
	train_labels_path=train_labels,
	test_images_path=test_images,
	test_labels_path=test_labels
)

(x_train, y_train), (x_test, y_test) = data_loader.load()


def plot_image(index: int):
	plt.figure()
	plt.imshow(x_test[index])  # , cmap=plt.cm.binary
	plt.title(y_test[index], fontsize=25)
	plt.colorbar()
	plt.show()


plot_image(28)



