import keras
import numpy as np
import matplotlib.pyplot as plt

from train.loader import MnistDataloader


model = keras.models.load_model('../models/mnist-0.9815.keras')
probability_model = keras.Sequential([model, keras.layers.Softmax()])

# remove
test_images_path = '../data/t10k-images.idx3-ubyte'
test_labels_path = '../data/t10k-labels.idx1-ubyte'

data_loader = MnistDataloader(
	train_images_path=None,
	train_labels_path=None,
	test_images_path=test_images_path,
	test_labels_path=test_labels_path
)
#---------------

test_images, test_labels = data_loader.load_test()
image_to_predict = test_images[567]
arr_to_predict = np.array([image_to_predict])

prediction = probability_model.predict(arr_to_predict)
predict_arg = np.argmax(prediction)
# print(probability_model.predict(np.array(test_images[0])))
print(predict_arg)


# read radnom image
# image = cv.imread(file, cv.IMREAD_GRAYSCALE)
# image = cv.resize(image, (28, 28)) # ??
# image = image.astype('float32')
# image = image.reshape(1, 28, 28, 1)
# image = 255-image
# image /= 255
#----


def plot_image(image, label):
	plt.figure()
	plt.imshow(image)  # , cmap=plt.cm.binary
	plt.title(label, fontsize=20)
	plt.colorbar()
	plt.show()


plot_image(image_to_predict, predict_arg)

