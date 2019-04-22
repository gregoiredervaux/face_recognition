import cv2
import numpy as np
from models import face_track_server, face_describer_server, nn, camera_server
from configs import configs

'''
The demo app utilize all servers in model folder with simple business scenario/logics:
I have a camera product and I need to use it to find all visitors in my store who came here before.

Main logics is in the process function, where you can further customize.
'''


class Demo(camera_server.CameraServer):

    def __init__(self, *args, **kwargs):
        super(Demo, self).__init__(*args, **kwargs)
        self.face_tracker = face_track_server.FaceTrackServer()
        self.face_describer = face_describer_server.FDServer(
            model_fp=configs.face_describer_model_fp,
            input_tensor_names=configs.face_describer_input_tensor_names,
            output_tensor_names=configs.face_describer_output_tensor_names,
            device=configs.face_describer_device)
        self.nn_model = nn.Model(path_to_model="../pretrained/init_custom.hdf5")

    def processs(self, frame):
        self.face_tracker.process(frame)
        _faces = self.face_tracker.get_faces()
        _num_faces = len(_faces)

        if _num_faces == 0:
            return
        for _face in _faces:
            cv2.imshow("img", _face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            _face_resize = cv2.resize(_face, configs.face_describer_tensor_shape)
            _data_feed = [np.expand_dims(_face_resize.copy(), axis=0), configs.face_describer_drop_out_rate]
            _face_description = self.face_describer.inference(_data_feed)[0][0]

            with self.nn_model.new_Graph.as_default():
                Y = self.nn_model.model.predict(np.matrix(_face_description))
                print(Y[0])
                print(np.argmax(Y[0]))

if __name__ == '__main__':

    demo = Demo(camera_address=0)
    demo.run()

