import time
import numpy as np

class HelperFunctions:
    @staticmethod
    def get_timestamp():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def normalize(value, min_val, max_val):
        if max_val - min_val == 0:
            return 0
        return (value - min_val) / (max_val - min_val)

    @staticmethod
    def calculate_attention_score(states):
        """
        states example:
        ['engaged','engaged','distracted','engaged']
        """
        total = len(states)
        engaged = states.count("engaged")

        if total == 0:
            return 0

        score = (engaged / total) * 100
        return round(score, 2)

    @staticmethod
    def moving_average(data, window=5):
        if len(data) < window:
            return data

        result = []
        for i in range(len(data)):
            start = max(0, i-window+1)
            avg = np.mean(data[start:i+1])
            result.append(avg)

        return result

    @staticmethod
    def print_system_status():
        print("="*40)
        print("Aqlli sinf diqqat tahlilchisi ishga tushdi")
        print("Vaqt:", HelperFunctions.get_timestamp())
        print("="*40)