from mtcnn import MTCNN
import cv2
import os
import numpy as np
import sys

class FaceAligner:

    def __init__(self, left_eye=(0.35, 0.35), width=256, height=None) -> None:
        self.left_eye = left_eye
        self.width = width
        if height is None:
            self.height = self.width
        else:
            self.height = height

    def align(self, image, points):
        left_center = points['left_eye']
        right_center = points['right_eye']
        dy = right_center[1] - left_center[1]
        dx = right_center[0] - left_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        right_eye = 1 - self.left_eye[0]

        dist = np.sqrt(dx ** 2 + dy ** 2)
        desired_dist = (right_eye - self.left_eye[0])
        desired_dist *= self.width
        scale = desired_dist / dist

        center = (left_center[0] + right_center[0]) // 2, (left_center[1] + right_center[1]) // 2
        M = cv2.getRotationMatrix2D(center, angle, scale)

        tx = self.width * 0.5
        ty = self.height * self.left_eye[1]
        M[0, 2] += (tx - center[0])
        M[1, 2] += (ty - center[1])

        w, h = self.width, self.height
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        return output

def main(dir_path, n_cpus, index):
    detector = MTCNN()
    fa = FaceAligner()

    path = os.path.join(dir_path, "data", "UTKFace")
    new_dir = os.path.join(dir_path, "data", "UTKFace", "aligned")
    for i, img_path in enumerate(os.listdir(path)):
        if i % n_cpus == index:
            old_path = os.path.join(path, img_path)
            image = cv2.cvtColor(cv2.imread(old_path), cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(image)
            new_path = os.path.join(new_dir, img_path)

            # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
            if len(result) > 0:
                bounding_box = result[0]['box']
                keypoints = result[0]['keypoints']

                new_img = fa.align(image, keypoints)

                cv2.imwrite(new_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
                print(i)
            else:
                cv2.imwrite(new_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                print(i, "No face detected!")

if __name__ == "__main__":
    dir_path = sys.argv[1]
    n_cpus = int(sys.argv[2])
    index = int(sys.argv[3]) - 1
    main(dir_path, n_cpus, index)