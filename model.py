import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


def main():

	print("In main")
	images, measurements = GetImagesAndMesurements()
	augmented_images, augmented_measurements = AugmentImages(images,measurements)

	print(len(images))
	print(len(measurements))

	print(len(augmented_images))
	print(len(augmented_measurements))

	x_train = np.array(augmented_images)
	y_train = np.array(augmented_measurements)
	
	RunAndSaveModel(x_train,y_train)

	K.clear_session()

	return

def GetImagesAndMesurements():

	lines = []

	with open('data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)

		for line in reader:
			lines.append(line)

	images = []
	measurements = []

	for line in lines:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = 'data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

	return images,measurements

def AugmentImages(images,measurements):
	augmented_images, augmented_measurements = [], []

	for image,measurement in zip(images,measurements):
		augmented_images.append(image)
		augmented_measurements.append(measurement)
		augmented_images.append(cv2.flip(image,1))
		augmented_measurements.append(measurement*-1.0)

	return augmented_images, augmented_measurements

def RunAndSaveModel(x_train,y_train):
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	model.fit(x_train,y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

	model.save('model.h5')

	return

if __name__ == '__main__':
    main()


