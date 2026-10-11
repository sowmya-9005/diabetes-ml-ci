import unittest
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_valid_prediction_returns_200(self):
        response = self.client.post(
            "/predict",
            json={
                "age": 0.038, "sex": 0.050, "bmi": 0.061,
                "bp": 0.021, "s1": -0.044, "s2": -0.034,
                "s3": -0.043, "s4": -0.002, "s5": 0.019,
                "s6": -0.017
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.get_json())

    def test_prediction_is_numeric(self):
        response = self.client.post(
            "/predict",
            json={
                "age": 0.038, "sex": 0.050, "bmi": 0.061,
                "bp": 0.021, "s1": -0.044, "s2": -0.034,
                "s3": -0.043, "s4": -0.002, "s5": 0.019,
                "s6": -0.017
            }
        )
        result = response.get_json()
        self.assertIsInstance(result["prediction"], float)

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"age": 0.038, "bmi": 0.061}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
