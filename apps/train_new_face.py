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
        # Step1. Find and track face (frame ---> [Face_Tracker] ---> Faces Loactions)
        self.face_tracker.process(frame)
        _faces = self.face_tracker.get_faces()

        # Uncomment below to visualize face
        #_faces_loc = self.face_tracker.get_faces_loc()
        #self._viz_faces(_faces_loc, frame)

        # Step2. For each face, get the cropped face area, feeding it to face describer (insightface) to get 512-D Feature Embedding
        _face_descriptions = []
        _num_faces = len(_faces)
        if _num_faces == 0:
            return
        for _face in _faces:
            # Step3. For each face, check whether there are similar faces and if not save it to db.
            # Below naive and verbose implementation is to tutor you how this work
            dir = os.listdir(configs.db_custom_path + self.name)
            cv2.imwrite(configs.db_custom_path + self.name + "/" + str(len(dir) + 1) + ".jpg", _face)


    def run(self):
        print('[Camera Server] Camera is initializing ...')
        if self.camera_address is not None:
            self.cam = cv2.VideoCapture(self.camera_address)
        else:
            print('[Camera Server] Camera is not available!')
            return

        while len(os.listdir(configs.db_custom_path + self.name)) <= 150:
            self.in_progress = True

            # Grab a single frame of video
            ret, frame = self.cam.read()
            self.processs(frame)
        self.nn_model.add_class()
        data = dataloader.DataLoader(self.face_describer)
        data.load_class_data(configs.db_custom_path + self.name, self.nn_model.get_nb_classes() - 1)
        data.serialyse(configs.model_pretrained_path, self.name)
        self.nn_model.train_model_from_data(np.array(data.X), np.array(data.Y), epoch=20)


if __name__ == '__main__':

    train = TrainNewFace(sys.argv[1], camera_address=0)
    train.run()

