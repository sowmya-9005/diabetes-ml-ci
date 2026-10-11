import json
import os
import unittest
import joblib
import numpy as np
from sklearn.datasets import load_diabetes


class TestMLPipeline(unittest.TestCase):

    def test_model_created(self):
        self.assertTrue(os.path.exists("diabetes_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_r2_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)
        r2 = metrics["r2"]
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2, 1.0)

    def test_model_prediction(self):
        model = joblib.load("diabetes_model.pkl")
        data = load_diabetes()
        sample = data.data[:1]
        prediction = model.predict(sample)[0]
        self.assertGreater(prediction, 0)

    def test_high_risk_patient(self):
        model = joblib.load("diabetes_model.pkl")
        sample = np.array([[0.05, 0.05, 0.06, 0.04, 0.05, 0.05, 0.05, 0.03, 0.06, 0.05]])
        prediction = model.predict(sample)[0]
        self.assertGreater(prediction, 0)

    def test_low_risk_patient(self):
        model = joblib.load("diabetes_model.pkl")
        sample = np.array([[-0.05, -0.05, -0.06, -0.04, -0.05, -0.05, -0.05, -0.03, -0.06, -0.05]])
        prediction = model.predict(sample)[0]
        self.assertGreater(prediction, 0)


if __name__ == "__main__":
    unittest.main()
