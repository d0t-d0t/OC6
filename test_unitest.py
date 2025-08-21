import unittest
import main as app

class Test_Global(unittest.TestCase):
    def test_pipeline_import(self):
        self.assertIsNotNone(app.pipeline)

    # def test_model_prediction(self):
    #     prediction = int(app.latest_model.predict('hello world')[0])
    #     self.assertIn([0,1])

if __name__ == '__main__':
    unittest.main()