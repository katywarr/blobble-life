import unittest
from gym_blobble.BlobbleConfig import BlobbleConfig


class TestBlobbleConfig(unittest.TestCase):

    def test_read_invalid_config(self):
        config = BlobbleConfig('testdata/blobble_config_est.ini')
        self.assertEqual(None, config.get_learning_params())

    def test_read_config_missing_section(self):
        try:
            config = BlobbleConfig('testdata/blobble_config_test_missing_section.ini')
        except KeyError:
            return

        assert()    # KeyError should have been returned

    def test_read_config_learning(self):
        try:
            config = BlobbleConfig('testdata/blobble_config_test.ini')
        except:
            assert()    # Configuration should be syntactically correct, so no exception

        # Check a value
        self.assertEqual(10, config.get_learning_params()['num_eval_episodes'])

    def test_read_config_learning_adv(self):
        try:
            config = BlobbleConfig('testdata/blobble_config_test.ini')
        except:
            assert()    # Configuration should be syntactically correct, so no exception

        # Check a value
        self.assertEqual('5e-3', config.get_learning_adv_params()['learning_rate'])

    def test_read_config_output(self):
        try:
            config = BlobbleConfig('testdata/blobble_config_test.ini')
        except:
            assert()    # Configuration should be syntactically correct, so no exception

        # Check a value
        self.assertEqual('True', config.get_output_params()['demonstration_video'])

    def test_print_config(self):
        try:
            config = BlobbleConfig('testdata/blobble_config_test.ini')
        except:
            assert ()  # Configuration should be syntactically correct, so no exception

        config.print_config()


if __name__ == '__main__':
    unittest.main()
