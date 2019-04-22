import cv2
import numpy as np
from models import face_track_server, face_describer_server, nn, camera_server, dataloader
from configs import configs
import os
import sys

'''
The demo app utilize all servers in model folder with simple business scenario/logics:
I have a camera product and I need to use it to find all visitors in my store who came here before.

Main logics is in the process function, where you can further customize.
'''


class TrainNewFace(camera_server.CameraServer):

    def __init__(self, name, *args, **kwargs):
        super(TrainNewFace, self).__init__(*args, **kwargs)
        self.name = name
        self.face_tracker = face_track_server.FaceTrackServer()
        self.face_describer = face_describer_server.FDServer(
            model_fp=configs.face_describer_model_fp,
            input_tensor_names=configs.face_describer_input_tensor_names,
            output_tensor_names=configs.face_describer_output_tensor_names,
            device=configs.face_describer_device)
        self.nn_model = nn.Model(path_to_model="../pretrained/init.hdf5")
        try:
            os.mkdir(configs.db_custom_path + name)
        except:
            print(name + "existe déjà")


    def processs(self, frame):
        self.face_tracker.process(frame)
        _faces = self.face_tracker.get_faces()
        _face_descriptions = []
        _num_faces = len(_faces)

        if _num_faces == 0:
            return
        for _face in _faces:

            dir = os.listdir(configs.db_custom_path + self.name)
            cv2.imwrite(configs.db_custom_path + self.name + "/" + str(len(dir) + 1) + ".jpg", frame)


    def run(self):
        print('[Camera Server] Camera is initializing ...')
        if self.camera_address is not None:
            self.cam = cv2.VideoCapture(self.camera_address)
        else:
            print('[Camera Server] Camera is not available!')
            return

        while len(os.listdir(configs.db_custom_path + self.name)) <= 100:
            self.in_progress = True
            ret, frame = self.cam.read()
            self.processs(frame)

        self.nn_model.add_class()
        data_custom = dataloader.DataLoader(self.face_describer)
        data_custom.load_class_data(configs.db_custom_path + self.name, self.nn_model.get_nb_classes() - 1, False)
        data_custom.serialyse(configs.model_pretrained_path, self.name)
        #data_custom.deserialyse(configs.model_pretrained_path, self.name)
        #data_custom.split_from_index(128)

        data_init = dataloader.DataLoader(self.face_describer)
        data_init.deserialyse(configs.model_pretrained_path)
        data_init.shuffle()
        #data_init.split_from_index(128)

        data_custom.X = data_init.X[:min(len(data_custom.X)*8, len(data_init.X))] + data_custom.X
        data_custom.Y = data_init.Y[:min(len(data_custom.Y)*8, len(data_init.X))] + data_custom.Y
        data_custom.shuffle()
        self.nn_model.train_model_from_data(np.array(data_custom.X), np.array(data_custom.Y), epoch=40)
        self.nn_model.save_model(configs.model_pretrained_path + "init_custom" + configs.save_model_format)

        with self.nn_model.new_Graph.as_default():
            loss, accuray = self.nn_model.model.evaluate(np.array(data_custom.X), np.array(data_custom.Y))
            print(loss)
            print(accuray)

if __name__ == '__main__':

    train = TrainNewFace(sys.argv[1], camera_address=0)
    train.run()

