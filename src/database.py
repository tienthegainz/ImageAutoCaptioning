"""
    Class for saving vector feature and saving word-embedded vector feature
    Using Pickle, saving under database folder
"""

import pickle


class DBManagement():
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if DBManagement.__instance is None:
            DBManagement()
        return DBManagement.__instance

    def __init__(self):
        """ Virtually private constructor. """
        self.data = None
        if DBManagement.__instance is not None:
            print('DB class is a singleton!')
            self.get_database('database/image_vector.pkl')
        else:
            DBManagement.__instance = self
            self.get_database('database/image_vector.pkl')

    def get_database(self, file):
        try:
            self.data = pickle.loads(open(file, "rb").read())
        except:
            return

    def save_image(self):
        f = open(database/image_vector.pkl, "wb+")
        f.write(pickle.dumps(self.data))
        f.close()

    def save_image_data(self, vector_dictionary):
        f = open('database/image_vector.pkl', "wb+")
        self.data = vector_dictionary
        f.write(pickle.dumps(self.data))
        f.close()

    def get_image_data(self):
        return self.data

    def get_image_data_from_list(self, data_path):
        # FIXME
        """
            data_path: path to train_list, test_list, val_list
        """
        path_data = dict()
        file = open(data_path, 'r')
        for line in file:
            line = line[0:-1]
            path_data[line] = self.data[line]
        return path_data
