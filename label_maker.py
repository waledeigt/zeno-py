# Code created by John Coxon 

import numpy as np
import matplotlib.gridspec as gridspec

def draw_labels(ax, xy = (-0.1,1.1), xytext = (0.5,-0.5)):
    """Draws labels on each subplot in an array with a GridSpec-compatible shape."""

    label_list = labels(ax.shape)
    print(label_list)

    for Cr, Vr in enumerate(ax):
        try:
            for Cc, Vc in enumerate(Vr):
                Vc.annotate(label_list[Cr,Cc], xy, xytext = xytext, xycoords = 'axes fraction',
                    textcoords = 'offset pixels', va = 'top', ha = 'left', size = 'x-large')
        except TypeError:
            Vr.annotate(label_list[Cr], xy, xytext = xytext, xycoords = 'axes fraction',
                textcoords = 'offset pixels', va = 'top', ha = 'left', size = 'x-large')

def labels(shape):
    """Returns a list of consecutive text labels starting at 'a' as an array in that shape."""
    length = np.prod(shape)
    arange = np.arange(0,length)
    labels = np.zeros(length,dtype='unicode_')
    string = 'abcdefghijklmnopqrstuvwxyz'

    for i in arange:
        labels[i] = string[i]

    labels = np.reshape(labels, shape)

    return(labels)