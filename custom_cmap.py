# code adapted from https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale by unutbu

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_cmap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1). For example: make_cmap([c('white'), c('cyan'), 0.10, c('cyan'), c('blue'), 0.50, c('blue'),\
                                      c('lime'), 0.90, c('lime')]) {the color map used in Weigt et al. (in prep.)
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,)*3]
    cdict = {'red':[], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
        
    