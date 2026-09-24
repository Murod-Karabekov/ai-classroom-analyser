import numpy as np
import cv2

class HeadPoseEstimator:
    def __init__(self):
        # approximate 3D face model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose
            (0.0, -330.0, -65.0),   # Chin
            (-225.0, 170.0, -135.0),# Left eye
            (225.0, 170.0, -135.0), # Right eye
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

    def estimate_pose(self, landmarks, frame_shape):
        h, w = frame_shape[:2]

        focal_length = w
        center = (w/2, h/2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0,0,1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))

        image_points = np.array([
            landmarks[1],   # nose
            landmarks[152], # chin
            landmarks[33],  # left eye
            landmarks[263], # right eye
            landmarks[61],
            landmarks[291]
        ], dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        return rotation_vector

    def is_looking_away(self, rotation_vector, threshold=None):
        from src.utils.config import Config
        rotation_magnitude = np.linalg.norm(rotation_vector)
        return 1 if rotation_magnitude > Config.HEAD_POSE_THRESHOLD else 0  


if __name__ == "__main__":
    estimator = HeadPoseEstimator()
    fake_landmarks = [(0,0)]*468

    pose = estimator.estimate_pose(fake_landmarks,(480,640,3))
    print("Rotation vector:", pose)