import numpy as np
import pandas as pd
import mne
import os
import json
from abc import ABC, abstractmethod



class AbstractConstructor(ABC):
    def __init__(self, config):
        self.config = config
        super().__init__()

    @abstractmethod
    def create_coords_config(self):
        pass

    @abstractmethod
    def map_coords(self, df, coords_config):
        pass


class ImgRepresentationCreator(AbstractConstructor):
    """
    Class for preparing an instruction about
    how to place signals into the image
    """
    def __init__(self, config):
        """
        __init__
        :param config: config in json format
        """
        super().__init__(config)
        self.img_width = self.config["param"]["img_width"]
        self.img_height = self.config["param"]["img_height"]
        self.coords_dict = None

    def _get_desc_values(self, locs, center_sensor_id):
        """
        Calculates several values, that important
        for calculation of the coordinates of sensors
        :param locs: dict with sensors numbers
            corresponding to their relative position
        :param center_sensor_id: number corresponding
            to the name of senter sensor
        :return:
            sensors_width:
            sensors_height: number of rows with sensors
                in the resulting image
            center_row_num: rows number, that corresponds
                to the center sensor (the one, that
                should be in the middle of the image)
        """
        sensors_height = len(locs)
        sensors_width = 0
        center_row_num = 0
        for i in range(len(locs)):
            rowkey = "row_{}".format(i+1)
            cur_row = locs[rowkey]
            if sensors_width < len(cur_row):
                sensors_width = len(cur_row)
            if center_sensor_id in cur_row:
                center_row_num = i + 1
        return sensors_width, sensors_height, center_row_num

    def get_2D_coords_from_config(self):
        """
        Creates a dict with names of sensors
        as keys and lists of coordinates as values
        :return: dict with coordinates
        """
        coords_dict = {}
        if self.config["param"]["type"]=="2d_manual":
            # Reading parametrs from config
            # Creating parametrs for calculations
            center_sensor_id = self.config["param"]["center_sensor_id"]
            img_width_center = self.img_width // 2
            img_height_center = self.img_height // 2
            row_step = 1 + np.int(self.config["param"]["zero_pad_row"]) + \
                       2 * np.int(self.config["param"]["repeat_pad_row"])
            # col_step defines the zero padding for columns of image
            col_step = 1 + np.int(self.config["param"]["zero_pad_col"]) + \
                       2 * np.int(self.config["param"]["repeat_pad_col"])
            locs = self.config["loc"]
            sensors_width, sensors_height, center_row_num = self._get_desc_values(locs, center_sensor_id)

            img_height_cells = sensors_height * row_step
            img_width_cells = sensors_width * col_step
            if img_height_cells>self.img_height:
                raise ValueError("Not enough height for this number of rows of sensors")
            if img_width_cells>self.img_width:
                raise ValueError("Not enough width for this number of rows of sensors")

            # Going through rows and defining coordinates for each sensor in each row
            for i in range(len(locs)):
                rowkey = "row_{}".format(i + 1)
                cur_row = locs[rowkey]
                y_coord = img_height_center + row_step * (i + 1 - center_row_num)
                # Defining the row width based on the usage
                # (or non usage) of zero padding for columns
                row_width = len(cur_row)
                if self.config["param"]["repeat_pad_col"]:
                    row_width = row_width * (2 + 1)
                if self.config["param"]["zero_pad_col"]:
                    row_width = row_width + len(cur_row) - 1
                # Going through sensors in the row
                start_x_coord = img_width_center - row_width // 2
                for j in range(len(cur_row)):
                    x_coord = start_x_coord + col_step * j
                    sensor_name = "EEG_{}".format(cur_row[j])
                    if sensor_name not in coords_dict.keys():
                        coords_dict[sensor_name] = [[x_coord, y_coord]]
                    else:
                        coords_dict[sensor_name].append([x_coord, y_coord])
                    # Adding surrounding coords
                    surrounding_coords = self.config["param"]["repeat_pad_col"] | \
                                         self.config["param"]["repeat_pad_row"]
                    if surrounding_coords:
                        sensor_surr_name = "EEG_{}_surr".format(cur_row[j])
                        coords_dict[sensor_surr_name] = []
                        if self.config["param"]["repeat_pad_col"]:
                            coords_dict[sensor_surr_name].append([x_coord + 1, y_coord])
                            coords_dict[sensor_surr_name].append([x_coord - 1, y_coord])
                        if self.config["param"]["repeat_pad_row"]:
                            coords_dict[sensor_surr_name].append([x_coord, y_coord + 1])
                            coords_dict[sensor_surr_name].append([x_coord - 1, y_coord + 1])
                            coords_dict[sensor_surr_name].append([x_coord + 1, y_coord + 1])
                            coords_dict[sensor_surr_name].append([x_coord, y_coord - 1])
                            coords_dict[sensor_surr_name].append([x_coord - 1, y_coord - 1])
                            coords_dict[sensor_surr_name].append([x_coord + 1, y_coord - 1])
        else:
            raise ValueError("Use the correct type of config (for example, 2d_manual)")

        self.coords_dict = coords_dict
        return coords_dict

    def get_coordinates_table(self):
        """
        Creates the table, that shows
        where each sensor is putted in the image
        :return: pandas DataFrame with names of
            sensors in corresponding places and
            dots in others
        """
        img_df = pd.DataFrame(index=range(self.img_height),
                              columns=range(self.img_width))
        img_df = img_df.fillna(".")
        if self.coords_dict is None:
            return img_df
        for k in self.coords_dict.keys():
            for coords in self.coords_dict[k]:
                x, y = coords
                img_df.iloc[y, x] = k
        return img_df

    def create_coords_config(self):
        coords_dict = self.get_2D_coords_from_config()
        return coords_dict

    def map_coords(self, df, coords_config):
        img_array = np.zeros((self.img_height, self.img_width, 3))
        for c in df.columns:
            values = df[c].values
            coords = coords_config[c]
            for coord in coords:
                x, y = coord
                img_array[y, x, 0] = values[0]
                img_array[y, x, 1] = values[1]
                img_array[y, x, 2] = values[2]
            if c + "_surr" in coords_config.keys():
                coords_surr = coords_config[c + "_surr"]
                for coord in coords_surr:
                    x, y = coord
                    img_array[y, x, 0] = values[0]
                    img_array[y, x, 1] = values[1]
                    img_array[y, x, 2] = values[2]
        return img_array



if __name__=="__main__":

    print()
