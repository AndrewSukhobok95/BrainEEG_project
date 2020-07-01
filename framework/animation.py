import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from state_clf.data_prep.constructor import ImgRepresentationCreator
from state_clf.data_prep.data_reader import DataReaderMatLab
from state_clf.data_prep.transformer import FFTtoPCA_transformer
from state_clf.data_prep.pipeline import PipeLine



def animate_brain(pipeline, n_frames=100):
    fig = plt.figure()
    ims = []
    for i in range(n_frames):
        obs, y = pipeline.get_obs(i)
        # obs = np.abs(obs * 1 / obs.max())
        im = plt.imshow(obs, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims,
                                    interval=50, blit=False,
                                    repeat_delay=0)
    plt.show()



if __name__=="__main__":
    with open("./configs/2d_manual_125.json", 'r') as f:
        config = json.load(f)

    constructor_irc = ImgRepresentationCreator(config)
    data_reader = DataReaderMatLab("./../data/EEG tubingen/TMS-EEG_Tubingen_ver2.mat")
    transformer = FFTtoPCA_transformer(pca_models_name="PCAmodel_clf1s")

    img_df = constructor_irc.get_coordinates_table()

    pipeline = PipeLine(
        constructor=constructor_irc,
        data_reader=data_reader,
        transformer=transformer)

    # pipeline.fit_transformers()

    animate_brain(pipeline)

    print()

