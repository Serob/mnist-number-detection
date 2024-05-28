import numpy as np
import struct


class MnistDataloader(object):
	def __init__(self, train_images_path, train_labels_path, test_images_path, test_labels_path):
		self.training_images_path = train_images_path
		self.training_labels_path = train_labels_path
		self.test_images_path = test_images_path
		self.test_labels_path = test_labels_path

	@staticmethod
	def read_images_labels(images_path, labels_path):
		labels = []
		with open(labels_path, 'rb') as file:
			magic, size = struct.unpack(">II", file.read(8))
			if magic != 2049:
				raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
			labels = np.frombuffer(file.read(), dtype=np.uint8)

		with open(images_path, 'rb') as file:
			magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
			if magic != 2051:
				raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
			images = np.frombuffer(file.read(), dtype=np.uint8).reshape(size, 28, 28)
		return images, labels

	def load_all(self):
		images_train, labels_train = self.read_images_labels(self.training_images_path, self.training_labels_path)
		images_test, labels_test = self.read_images_labels(self.test_images_path, self.test_labels_path)
		return (images_train, labels_train), (images_test, labels_test)

	def load_test(self):
		images_test, labels_test = self.read_images_labels(self.test_images_path, self.test_labels_path)
		return images_test, labels_test
