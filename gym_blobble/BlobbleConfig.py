import configparser
from pathlib import Path


class BlobbleConfig:
    """
    Class to store the configuration for the currently running BlobbleWorld.
    The configuration is either passed through parameters or els eread from a file.
    """

    def __init__(self,
                 config_file='blobble_config.ini'):


        config = configparser.ConfigParser()

        self._learning_params = None
        self._learning_adv_params = None
        self._output_params = None

        if not Path(config_file).exists():
            print("Error: File ", config_file, " does not exist.")
            return

        config.read(config_file)

        # Check for sections as expected is in the config file
        sections = ['LEARNING', 'LEARNING-ADVANCED', 'OUTPUT']
        for section in sections:
            if not config.has_section(section):
                print("Error: ", config_file, " is not of the correct format. It must contain all the sections. "
                                              "The following section is missing from the file: ", section)
                raise KeyError(section)

        string_dict_learning = dict(config['LEARNING'])
        self._learning_adv_params = dict(config['LEARNING-ADVANCED'])
        self._output_params = dict(config['OUTPUT'])

        try:
            # Learning parameters are all integers, so we can change the type of all the values from string
            # to make things simpler later on.
            self._learning_params = dict((k, int(v)) for k, v in string_dict_learning.items())
        except ValueError as err:
            print("Error: ", config_file, " is not of the correct format. The value of one of the parameters in the "
                                          "LEARNING section is incorrect. Here's the error: ", err)
            raise err

    def print_config(self):

        print('\n=====================================================')
        print('       BlobbleWorld Configuration')
        print('       --------------------------')
        for k, v in self._learning_params.items():
            print(k, ' : ', v)
        for k, v in self._learning_adv_params.items():
            print(k, ' : ', v)
        for k, v in self._output_params.items():
            print(k, ' : ', v)
        print('=====================================================')

    def get_learning_params(self):
        return self._learning_params

    def get_learning_adv_params(self):
        return self._learning_adv_params

    def get_output_params(self):
        return self._output_params
