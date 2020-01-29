import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class AbstractPipeLine(ABC):
    def __init__(self,
                 constructor,
                 data_reader,
                 transformer):
        self.constructor = constructor
        self.data_reader = data_reader
        self.transformer = transformer
        self.coords_config = self.constructor.create_coords_config()
        super().__init__()

    @abstractmethod
    def get_number_of_obs(self):
        pass

    @abstractmethod
    def get_obs(self, index):
        pass



class PipeLine(AbstractPipeLine):

    def get_number_of_obs(self):
        return self.data_reader.get_number_of_obs()

    def fit_transformers(self):
        sample_data_dict = self.data_reader.random_sensor_sample()
        self.transformer.fitPCA(sample_data_dict)

    def get_obs(self, index):
        obs, y = self.data_reader.get_obs(index=index)
        transformed_obs = self.transformer.transform(obs)
        wave_img = self.constructor.map_coords(transformed_obs, self.coords_config)
        return wave_img, y




if __name__=="__main__":

    print()
