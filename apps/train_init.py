import numpy as np
from models import face_describer_server, nn, dataloader, face_track_server
from configs import configs
import cv2


'''
the module is used to  build the initial model and to save the initial 
'''
if __name__ == '__main__':
    face_describer = face_describer_server.FDServer(
        model_fp=configs.face_describer_model_fp,
        input_tensor_names=configs.face_describer_input_tensor_names,
        output_tensor_names=configs.face_describer_output_tensor_names,
        device=configs.face_describer_device)
    face_tracker = face_track_server.FaceTrackServer()
    data = dataloader.DataLoader(face_describer)


    data.deserialyse(configs.model_pretrained_path)
    #data.load_data(configs.db_path)
    #data.serialyse(configs.model_pretrained_path)
    data.shuffle()

    nn_model = nn.Model(output_shape=len(np.unique(np.array(data.Y))))
    #nn_model = nn.Model(path_to_model="../pretrained/init.hdf5")


    #data.split_from_index(128)
    nn_model.train_model_from_data(np.array(data.X), np.array(data.Y), 500)
    nn_model.save_model(configs.model_pretrained_path + "init" + configs.save_model_format)
    nn_model.model.summary()


    with nn_model.new_Graph.as_default():
        loss, test_accuracy = nn_model.model.evaluate(np.array(data.X[:1000]), np.array(data.Y[:1000]))
        print(test_accuracy)