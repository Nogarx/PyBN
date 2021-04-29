import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
from collections import Counter
from math import ceil

import time

plt.style.use('seaborn')
color = 'viridis'
label_size = 16
title_font_size = 16
plt.rcParams.update({'font.size': label_size})
tick_fontsize = 12
legend_font_size = 14

####################################################################################################
####################################################################################################
####################################################################################################
    
def box_plot(data_list, fig_size=(10,10), columns=2, figure_path=None):

    # Create figure.
    fig = plt.figure(figsize=fig_size)
    fig_spec = gridspec.GridSpec(ncols=columns, nrows=ceil(len(data_list) / columns), figure=fig)

    # Plot data.
    iterator = 0
    for data in data_list:
        figure_ax = fig.add_subplot(fig_spec[iterator // columns, iterator % columns])
        figure_ax.grid(False)
        figure_ax.imshow(data, cmap=color)
        iterator += 1

    # Save plot.
    if figure_path is not None:
        plt.savefig(figure_path + '.png')


####################################################################################################
####################################################################################################
####################################################################################################