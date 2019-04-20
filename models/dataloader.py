import numpy as np
import cv2
from configs import configs
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

class DataLoader:

    def __init__(self, face_describer):
        self.face_describer = face_describer
        self.image_generator = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=(0.7, 1.3),
            featurewise_center=False
        )
        self.X = []
        self.Y = []

    def load_data(self, path_to_data):
        dir_list = os.listdir(path_to_data)
        for directory in dir_list:
            index = dir_list.index(directory)
            self.load_class_data(path_to_data + directory, index)


    def load_class_data(self, db_class_path, class_index):
        X = []
        Y = []
        for filename in os.listdir(db_class_path):
            print(db_class_path + "/" + filename)
            img = cv2.imread(db_class_path + "/" + filename)
            X.append(self.getFeatures(img))
            Y.append(class_index)
            img_augmented_array = self.image_generator.flow(np.array([img]))
            i = 0
            for img_aug in img_augmented_array:
                X.append(self.getFeatures(img_aug[0]/255))
                Y.append(class_index)
                i += 1
                if i > 10:
                    break
        self.X += X
        self.Y += Y

    def getFeatures(self, img):
        img_resize = cv2.resize(img, configs.face_describer_tensor_shape)
        data_feed = [np.expand_dims(img_resize.copy(), axis=0), configs.face_describer_drop_out_rate]
        face_description = self.face_describer.inference(data_feed)[0][0]

        return face_description

    def serialyse(self, path, suffix=""):
        np.savetxt(path + "X" + suffix + ".txt", self.X)
        np.savetxt(path + "Y" + suffix + ".txt", self.Y)

    def deserialyse(self, path, suffix=""):
        self.X = np.loadtxt(path + "X" + suffix + ".txt")
        self.Y = np.loadtxt(path + "Y" + suffix + ".txt").astype(int)