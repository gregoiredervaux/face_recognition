import os

BASE_PATH = '/'.join(os.getcwd().split('/')[:-1]) # Using ubuntu machine may require removing this -1
face_describer_input_tensor_names = ['img_inputs:0', 'dropout_rate:0']
face_describer_output_tensor_names = ['resnet_v1_50/E_BN2/Identity:0']
face_describer_device = '/cpu:0'
face_describer_model_fp = '{}/pretrained/insightface.pb'.format(BASE_PATH)
face_describer_tensor_shape = (112, 112)
face_describer_drop_out_rate = 0.5
test_img_fp = '{}/tests/test.jpg'.format(BASE_PATH)
db_path = '{}/db/'.format(BASE_PATH)
db_custom_path = '{}/db_custom/'.format(BASE_PATH)
model_pretrained_path = '{}/pretrained/'.format(BASE_PATH)
save_model_format = ".hdf5"

face_similarity_threshold = 800