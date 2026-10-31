import unittest
import pandas as pd
from model import DataValidationError, load_model, predict_one, prepare_features, validate_dataframe

class FraudModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.sample = {"amount": 1250.50, "transaction_type": "purchase",
                      "merchant_category": "grocery", "customer_age": 34,
                      "payment_method": "upi", "transaction_hour": 14}

    def test_single_prediction_contract(self):
        result = predict_one(self.model, self.sample)
        self.assertIn(result["prediction"], {"FRAUD", "LEGITIMATE"})
        self.assertIn(result["risk_level"], {"LOW", "MEDIUM", "HIGH"})
        self.assertGreaterEqual(result["fraud_probability"], 0)
        self.assertLessEqual(result["fraud_probability"], 1)

    def test_missing_column_rejected(self):
        with self.assertRaises(DataValidationError):
            validate_dataframe(pd.DataFrame([{"amount": 10}]))

    def test_batch_feature_preparation(self):
        frame = pd.DataFrame([self.sample, self.sample])
        features = prepare_features(frame)
        self.assertEqual(len(features), 2)

if __name__ == "__main__":
    unittest.main()
