import unittest
import main as api

class Test_Global(unittest.TestCase):
    def test_pipeline_import(self):
        self.assertIsNotNone(api.pipeline)

if __name__ == '__main__':
    unittest.main()