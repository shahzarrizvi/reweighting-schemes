def joke():
    return (u'Wenn ist das Nunst\u00fcck git und Slotermeyer? Ja! ... '
            u'Beiherhund das Oder die Flipperwaldt gersput.CHANGE CHANGE CHANGE BLAH')

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib import rc
import matplotlib.gridspec as gridspec

import numpy as np
import math
import pickle
import itertools

import awkward
import uproot
import uproot3_methods

from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

from .plot import *
from .table import *
