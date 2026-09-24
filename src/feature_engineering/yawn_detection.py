import numpy as np

class YawnDetector:
    def __init__(self, threshold=None):
        if threshold is None:
            from src.utils.config import Config
            threshold = Config.YAWN_THRESHOLD

        self.threshold = threshold

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def mouth_aspect_ratio(self, mouth_points):
        A = self.distance(mouth_points[2], mouth_points[8])
        B = self.distance(mouth_points[4], mouth_points[5])
        C = self.distance(mouth_points[0], mouth_points[6])

        mar = (A + B) / (2.0 * C)
        return mar

    def detect_yawn(self, mouth_points):
        mar = self.mouth_aspect_ratio(mouth_points)

        if mar > self.threshold:
            return 1
        else:
            return 0


if __name__ == "__main__":
    mouth = [(10,10)]*12
    
    detector = YawnDetector()
    result = detector.detect_yawn(mouth)
    print("Yawn Detected:", result)