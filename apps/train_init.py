import numpy as np
from models import face_describer_server, nn, dataloader, face_track_server
from configs import configs
import cv2


'''
The demo app utilize all servers in model folder with simple business scenario/logics:
I have a camera product and I need to use it to find all visitors in my store who came here before.

Main logics is in the process function, where you can further customize.
'''
if __name__ == '__main__':
    face_describer = face_describer_server.FDServer(
        model_fp=configs.face_describer_model_fp,
        input_tensor_names=configs.face_describer_input_tensor_names,
        output_tensor_names=configs.face_describer_output_tensor_names,
        device=configs.face_describer_device)
    face_tracker = face_track_server.FaceTrackServer()
    data = dataloader.DataLoader(face_describer)
    img = cv2.imread("../tests/jennifer-lawrence_gettyimages-626382596jpg.jpg")
    face_tracker.process(img)
    test_face = face_tracker.get_faces()[0]
    X_test = data.getFeatures(test_face)

    #data.deserialyse(configs.model_pretrained_path)
    data.load_data(configs.db_path)
    data.serialyse(configs.model_pretrained_path)
    nn_model = nn.Model(output_shape=len(np.unique(data.Y)))
    #nn_model = nn.Model(path_to_model="../pretrained/init.hdf5")
    nn_model.train_model_from_data(data.X, data.Y)
    nn_model.save_model(configs.model_pretrained_path + "init" + configs.save_model_format)
    nn_model.model.summary()

    with nn_model.new_Graph.as_default():
        Y = nn_model.model.predict(np.matrix(X_test))
        print(np.argmax(Y[0]))

