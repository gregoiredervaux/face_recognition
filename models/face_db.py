'''
A dummy db storaing faces in memory
Feel free to make it fancier like hooking with postgres or whatever
This model here is just for simple demo app under apps
Don't use it for production dude.
'''
from services import face_services
import numpy as np
import cv2
import scipy
from models import face_track_server, face_describer_server
from configs import configs
import os
import keras
import tensorflow as tf
from keras import backend as K


class Model(object):

    faces = []
    faces_discriptions = []

    def __init__(self, face_tracker, face_describer):
        self.face_tracker = face_tracker
        self.face_describer = face_describer
        self.dir_list = [dir for dir in os.listdir(configs.db_path)[:2]]
        self.new_Graph = tf.Graph()
        with self.new_Graph.as_default():
            self.tfSess = tf.Session(graph=self.new_Graph)
        self.load_data()
        self.train_model()

    def getFeatures(self, img):
        self.face_tracker.process(img)
        if len(self.face_tracker.get_faces()) == 0:
            raise Exception("pas de visage trouvé")

        face = self.face_tracker.get_faces()[0]
        img_resize = cv2.resize(face, configs.face_describer_tensor_shape)
        data_feed = [np.expand_dims(img_resize.copy(), axis=0), configs.face_describer_drop_out_rate]
        face_description = self.face_describer.inference(data_feed)[0][0]

        return face_description

    def train_model(self):

        print("len(os.listdir(configs.db_path)) " + str(len(self.dir_list)))

        with self.new_Graph.as_default():
            self.model = keras.Sequential([
                keras.layers.Dense(128, activation=tf.nn.relu, name="hidden"),
                keras.layers.Dense(int(len(self.dir_list)), activation=tf.nn.softmax, name="output")
            ])
            self.model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
            self.X = np.array(self.X)
            self.Y = np.array(self.Y)
            print(self.X.shape, self.Y.shape)

            if os.path.isfile(configs.weight_save_path):
                self.model.fit(self.X, self.Y, epochs=0)
                self.model.load_weights(configs.weight_save_path)
            else:
                self.model.fit(self.X, self.Y)
                self.model.save_weights(configs.weight_save_path)


    def load_data(self):
        self.X = []
        self.Y = []
        for directory in self.dir_list:
            index = self.dir_list.index(directory)
            X, Y = self.load_class_data(directory, index)
            self.X += X
            self.Y += Y

    def load_class_data(self, db_class_path, class_index):
        X = []
        Y = []
        for filename in os.listdir(configs.db_path + db_class_path):
            try:
                print(configs.db_path + db_class_path + "/" + filename)
                img = cv2.imread(configs.db_path + db_class_path + "/" + filename)
                X.append(self.getFeatures(img))
                Y.append(class_index)
            except Exception as error:
                pass

        return X, Y

    def add_class(self):
        with self.new_Graph.as_default():
            weights = self.model.get_layer("output").get_weights()
            weights[0] = np.array([np.append(weights_ligne, np.random.normal(0,0.01, 1)) for weights_ligne in weights[0]])
            weights[1] = np.append(weights[1], 0)
            self.model.summary()
            self.model.pop()
            self.model.summary()
            self.dir_list = [dir for dir in os.listdir(configs.db_path)[:2]]
            new_layer = keras.layers.Dense(int(len(self.dir_list)) + 1, activation=tf.nn.softmax, name="output")
            self.model.add(new_layer)
            self.model.get_layer("output").set_weights(weights)
            self.model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
            self.model.summary()

    def train_new_class(self, new_class_path):
        with self.new_Graph.as_default():
            X, Y = self.load_class_data(new_class_path, len(self.dir_list))
            self.model.fit(X, Y)
            self.model.save_weights(configs.weight_save_path)

    def add_face(self, face_img, face_description):
        self.faces.append(face_img)
        self.faces_discriptions.append(face_description)

    def drop_all(self):
        self.faces = []
        self.faces_discriptions = []

    def get_all(self):
        return self.faces, self.faces_discriptions

    def get_similar_faces(self, face_description):
        print('[Face DB] Looking for similar faces in a DataBase of {} faces...'.format(len(self.faces)))
        if len(self.faces) == 0:
            return []
        # Use items in Python 3*, below is by default for Python 2*
        similar_face_idx = face_services.compare_faces(self.faces_discriptions, face_description)
        similar_faces = np.array(self.faces)[similar_face_idx]
        num_similar_faces = len(similar_faces)
        print('[Face DB] Found {} similar faces in a DataBase of {} faces...'.format(num_similar_faces, len(self.faces)))
        return similar_faces
