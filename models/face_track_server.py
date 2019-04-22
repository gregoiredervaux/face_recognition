import cv2
from faced import FaceDetector
from faced.utils import annotate_image

'''
This server:
    Input: Camera frame
    Output: Relative locations for each face, with [(tr_x, tr_y, bl_x, bl_y)]

x1,y1 ------
|          |
|          |
|          |
--------x2,y2
'''


class FaceTrackServer(object):

    faces = []
    face_locations = []
    face_relative_locations = []
    cam_h = None
    cam_w = None
    camera_address = None

    def __init__(self, down_scale_factor=0.25):
        assert 0 <= down_scale_factor <= 1
        self.down_scale_factor = down_scale_factor
        self.face_detector = FaceDetector()

    def get_cam_info(self):
        return {'camera': {'width': self.cam_w, 'height': self.cam_h, 'address': self.camera_address}}

    def reset(self):
        self.face_relative_locations = []
        self.face_locations = []
        self.faces = []

    def process(self, frame):
        self.reset()
        self.cam_h, self.cam_w, _ = frame.shape
        # Resize frame of video to 1/4 size for faster face recognition processing

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_img = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        self.face_locations = self.face_detector.predict(rgb_img)
        # Display the results
        if len(self.face_locations) > 1:
            self.face_locations = []

        for x, y, w, h, _ in self.face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            x1 = int((x - int(w / 2))*(1 - 0.1))
            y1 = int((y - int(h / 2))*(1 - 0.1))
            x2 = int((x + int(w / 2))*(1 + 0.1))
            y2 = int((y + int(h / 2))*(1 + 0.1))

            _face_area = frame[y1:y2, x1:x2, :]

            if _face_area.size != 0:
                self.faces.append(_face_area)

        print('[FaceTracker Server] Found {} faces!'.format(len(self.faces)))
        return self.faces

    def get_faces_loc(self):
        return self.face_locations

    def get_faces(self):
        return self.faces