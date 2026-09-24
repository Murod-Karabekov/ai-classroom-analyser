import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.computer_vision.video_capture import VideoCaptureManager
from src.computer_vision.face_detection import FaceDetector
from src.ml_model.predict_attention import AttentionPredictor

from src.utils.helper_functions import HelperFunctions
from src.utils.config import Config

from src.feature_engineering.eye_aspect_ratio import EyeAspectRatio
from src.feature_engineering.yawn_detection import YawnDetector
from src.feature_engineering.head_pose_estimation import HeadPoseEstimator
from src.computer_vision.landmark_detection import LandmarkDetector

from reports.engagement_report import EngagementReport

def uzbek_state(state):
    return "DIQQATLI" if state == "engaged" else "CHALG'IGAN"


def main():
    HelperFunctions.print_system_status()
    Config.create_folders()
    video = VideoCaptureManager(Config.VIDEO_SOURCE)
    face_detector = FaceDetector()
    predictor = AttentionPredictor(Config.MODEL_PATH)
    report = EngagementReport()

    eye_detector = EyeAspectRatio()
    yawn_detector = YawnDetector()
    head_pose = HeadPoseEstimator()
    landmark_detector = LandmarkDetector()

    state_buffer = []

    import cv2

    while True:
        frame = video.get_frame()

        if frame is None:
            break

        faces = face_detector.detect_faces(frame)
        frame = face_detector.draw_faces(frame, faces)

        landmarks_list = landmark_detector.get_landmarks(frame)

        for landmarks in landmarks_list:
            # ---------- EYE ----------
            left_eye = [landmarks[i] for i in [33,160,158,133,153,144]]
            right_eye = [landmarks[i] for i in [362,385,387,263,373,380]]
            ear_left = eye_detector.calculate_ear(left_eye)
            ear_right = eye_detector.calculate_ear(right_eye)

            ear_avg = (ear_left + ear_right) / 2
            eye_closed = 1 if ear_avg < Config.EAR_THRESHOLD else 0

            # ---------- YAWN ----------
            mouth_points = [landmarks[i] for i in [78, 82, 13, 87, 84, 88, 308, 312, 14, 317, 314, 318]]
            mouth_ratio = yawn_detector.mouth_aspect_ratio(mouth_points)
            yawn = yawn_detector.detect_yawn(mouth_points)

            # ---------- HEAD POSE ----------
            rotation_vector = head_pose.estimate_pose(landmarks, frame.shape)
            head_turn = head_pose.is_looking_away(rotation_vector)
            rotation_magnitude = float((rotation_vector ** 2).sum() ** 0.5)

            # ---------- FEATURE VECTOR ----------
            feature_vector = [1, eye_closed, head_turn, yawn]
            state = predictor.predict(feature_vector)

            report.add_state(state)

            eye_status = "YUMILGAN" if eye_closed else "OCHIQ"
            head_status = "BURILGAN" if head_turn else "TO'G'RI"
            yawn_status = "ESNASH ANIQLANDI" if yawn else "ESNASH ANIQLANMADI"
            uzbek_prediction = uzbek_state(state)

            print(f"Ko'zlar: {eye_status} | EAR: {ear_avg:.3f}")
            print(f"Og'iz: {yawn_status} | MAR: {mouth_ratio:.3f}")
            print(f"Bosh holati: {head_status} | burilish qiymati: {rotation_magnitude:.3f}")
            print(f"Yakuniy baho: {uzbek_prediction}")

            # ---------- DISPLAY ----------
            overlay_lines = [
                f"HOLAT: {uzbek_prediction}",
                f"KO'ZLAR: {eye_status} (EAR {ear_avg:.3f})",
                f"OG'IZ: {yawn_status} (MAR {mouth_ratio:.3f})",
                f"BOSH: {head_status}",
            ]

            for line_number, text in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    text,
                    (20, 35 + line_number * 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 if line_number else 0.6,
                    (0,255,0) if state == "engaged" else (0,0,255),
                    2
                )

        video.show_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release() 
    report.generate_and_show()

if __name__ == "__main__":
    main()