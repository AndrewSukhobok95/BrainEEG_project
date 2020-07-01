import numpy as np
import pandas as pd
import mne
import scipy.io

from abc import ABC, abstractmethod


class AbstractDataReader(ABC):
    DFAULT_NUMBER_OF_SENSORS = 125
    DEFAULT_SENSOR_NAMES = ["EEG_{}".format(i+1) for i in range(DFAULT_NUMBER_OF_SENSORS)]

    def __init__(self, file_path,
                 sensor_names=DEFAULT_SENSOR_NAMES,
                 number_of_sensors=DFAULT_NUMBER_OF_SENSORS,
                 Fs=None, frame_length=None):
        self.file_path = file_path  # Path to file
        self.number_of_sensors = number_of_sensors
        self.sensor_names = sensor_names
        self.Fs = Fs  # Sampling Frequency
        self.frame_length = frame_length  # The length of the frame in seconds
        # (if you have a continuous signal)
        self.data, self.n_obs = self._reading_function(file_path)
        super().__init__()

    @abstractmethod
    def _reading_function(self, file_path):
        pass

    @abstractmethod
    def get_number_of_obs(self):
        pass

    @abstractmethod
    def get_obs(self, index):
        pass

    @abstractmethod
    def random_sensor_sample(self, n):
        pass



class DataReaderMatLab(AbstractDataReader):
    def _reading_function(self, file_path):
        mat = scipy.io.loadmat(file_path)
        data = mat["EEG"][0][0][15]
        n_obs = data.shape[2]
        return data, n_obs

    def get_number_of_obs(self):
        return self.n_obs * 2

    def get_obs(self, index):
        index_half = index // 2
        y_position = index % 2

        if y_position:
            df_e = pd.DataFrame(self.data[:, 1000:, index_half].T)
        else:
            df_e = pd.DataFrame(self.data[:, :1000, index_half].T)

        df_e.columns = self.sensor_names

        y = np.array([0,0])
        y[y_position] = 1

        return df_e, y

    def random_sensor_sample(self, n=1000):
        idx = np.random.randint(self.data.shape[2], size=n)
        sample_dict = {}
        sample_dict["sensor_names"] = self.sensor_names
        sample_dict["data"] = {}
        for i in range(self.number_of_sensors):
            sensor_sample = np.concatenate([self.data[0, :1000, idx].T, self.data[0, 1000:, idx].T], axis=1)
            sample_dict["data"][self.sensor_names[i]] = sensor_sample
        return sample_dict






if __name__ == "__main__":

    dr = DataReaderMatLab("./../../data/EEG tubingen/TMS-EEG_Tubingen_ver2.mat")
    data_dict = dr.random_sensor_sample()

    dr.get_obs(7)

    # 0 	 ( 1,)
    # 1 	 ( 1,)
    # 2 	 ( 1,)
    # 3 	 ( 0,)
    # 4 	 ( 0,)
    # 5 	 ( 0,)
    # 6 	 (0, 0)
    # 7 	 ( 0,)
    # 8 	 (1, 1)
    # 9 	 (1, 1)
    # 10 	 (1, 1)
    # 11 	 (1, 1)
    # 12 	 (1, 1)
    # 13 	 (1, 1)
    # 14 	 (1, 2000)
    # 15 	 (125, 2000, 1178)
    # 16 	 (0, 0)
    # 17 	 (125, 120)
    # 18 	 (125, 125)
    # 19 	 (120, 125)
    # 20 	 (1, 125)
    # 21 	 (1, 125)
    # 22 	 (0, 0)
    # 23 	 (1, 1)
    # 24 	 ( 1,)
    # 25 	 (0, 0)
    # 26 	 (0, 0)
    # 27 	 (0, 0)
    # 28 	 (0, 0)
    # 29 	 (0, 0)
    # 30 	 (1, 1)
    # 31 	 (1, 1)
    # 32 	 (0, 0)
    # 33 	 (0, 0)
    # 34 	 ( 0,)
    # 35 	 ( 0,)
    # 36 	 (0, 0)
    # 37 	 ( 1,)
    # 38 	 ( 1,)
    # 39 	 (1, 1)
    # 40 	 (1, 1)
    # 41 	 (1, 3)
    # 42 	 (1, 1)
    # 43 	 ( 1,)

    print()
