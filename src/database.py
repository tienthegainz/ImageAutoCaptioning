"""
    Class for saving vector feature and saving word-embedded vector feature
    Using Pickle, saving under database folder
"""

import pickle
import config


class DBManagement():
    __instance = None
    global data

    @staticmethod
    def get_instance():
        """ Static access method. """
        if DBManagement.__instance is None:
            DBManagement()
        return DBManagement.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if DBManagement.__instance is not None:
            print('DB class is a singleton!')
        else:
            DBManagement.__instance = self
            self.get_database(config.image_feature_file)

    def get_database(self, file):
        try:
            self.data = pickle.loads(open(file, "rb").read())
        except:
            return

    def save_image(self):
        f = open(config.image_feature_file, "wb+")
        f.write(pickle.dumps(self.data))
        f.close()

    def save_image_data(self, vector_dictionary):
        f = open(config.image_feature_file, "wb+")
        self.data = vector_dictionary
        f.write(pickle.dumps(self.data))
        f.close()

