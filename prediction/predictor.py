import keras
import numpy as np
import matplotlib.pyplot as plt


model = keras.models.load_model('../models/mnist-0.9815.keras')
probability_model = keras.Sequential([model, keras.layers.Softmax()])


def predict(image):
	arr_to_predict = np.array([image])
	prediction = probability_model.predict(arr_to_predict)
	predict_arg = np.argmax(prediction)
	if prediction[0][predict_arg] > 0.8:
		print(f'Prediction wight: {prediction[0][predict_arg]} for {predict_arg}')
		return [predict_arg]
	else:
		sorted_results = np.sort(prediction[0])
		second_important_index = np.argwhere(prediction[0] == sorted_results[-2])[0][0]
		print(f'Prediction wights: {prediction[0][predict_arg]} for {predict_arg} and {sorted_results[-2]} for {second_important_index}')
		return [predict_arg, second_important_index]


def plot_image(image, label):
	plt.figure()
	plt.imshow(image)  # , cmap=plt.cm.binary
	plt.title(label, fontsize=20)
	plt.colorbar()
	plt.show()
