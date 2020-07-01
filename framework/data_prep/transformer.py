import numpy as np
import pandas as pd
import pickle
import os
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod



class AbstractTransformer(ABC):
    def __init__(self, n_dim):
        self.n_dim = n_dim
        super().__init__()

    @abstractmethod
    def transform(self, df, Fs):
        pass



class PCAtransformation(object):
    MODEL_NAME = "PCAmodel"

    def __init__(self, models_name=MODEL_NAME):
        MODELS_FILE_PATH = "./../pretrained/pca/"
        models_path = os.path.abspath(
            os.path.normpath(
                os.path.join(__file__, "./../", MODELS_FILE_PATH)))
        self.models_path = models_path
        self.models_name = models_name
        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def fit(self, data_dict, n_components=3):
        sensor_names = data_dict["sensor_names"]
        model_dict = {}
        model_dict["sensor_names"] = sensor_names
        model_dict["models"] = {}
        for sn in sensor_names:
            X = data_dict["data"][sn]
            pca_model = PCA(n_components=n_components)
            pca_model.fit(X)
            model_dict["models"][sn] = pca_model
        pickle_path = '{}/{}.pickle'.format(self.models_path, self.models_name)
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_models(self, pickle_path=None):
        if pickle_path is None:
            pickle_path = '{}/{}.pickle'.format(self.models_path, self.models_name)
        with open(pickle_path, 'rb') as f:
            model_dict = pickle.load(f)
        return model_dict



class FFTtoPCA_transformer(AbstractTransformer):
    def __init__(self, pca_models_name="PCAmodel", n_dim=3):
        self.pca_model_dict = None
        self.pca_models_name = pca_models_name
        DEFAULT_PCA_PICKLE = "./../pretrained/pca/{}.pickle".format(self.pca_models_name)
        models_path = os.path.abspath(
            os.path.normpath(
                os.path.join(__file__, "./../", DEFAULT_PCA_PICKLE)))
        if os.path.exists(models_path):
            self.pca_model_dict = self._loadPCA()
        super().__init__(n_dim)

    def _get_spectrum(self, df):
        eps = 0.0000001
        n_points = df.shape[0]
        freqs = np.fft.fftfreq(n_points)
        mask = freqs > 0
        signal_fft = np.fft.fft(df.values, axis=0)[mask]
        powerspectrum = np.real(np.log(np.abs(np.fft.fft(signal_fft)) ** 2 + eps))
        df_powerspectrum = pd.DataFrame(powerspectrum, columns=df.columns)
        return df_powerspectrum

    def _loadPCA(self, pickle_path=None):
        pcat = PCAtransformation(models_name=self.pca_models_name)
        model_dict = pcat.load_models(pickle_path)
        return model_dict

    def _performPCA(self, df):
        transformed_df = pd.DataFrame()
        for sn in self.pca_model_dict["sensor_names"]:
            pca_model = self.pca_model_dict["models"][sn]
            transformed_df[sn] = pca_model.transform(df[sn].values.reshape(-1, 1).T)[0]
        return transformed_df

    def fitPCA(self, sensor_dict, n_components=3):
        for sn in sensor_dict["sensor_names"]:
            df_spectrum = self._get_spectrum(pd.DataFrame(sensor_dict["data"][sn]))
            sensor_dict["data"][sn] = df_spectrum.T
        pcat = PCAtransformation(models_name=self.pca_models_name)
        pcat.fit(sensor_dict, n_components)
        self.pca_model_dict = self._loadPCA()

    def transform(self, df, Fs=2000):
        spectrum = self._get_spectrum(df)
        pca_dots = self._performPCA(spectrum)
        return pca_dots





if __name__=="__main__":

    print()
