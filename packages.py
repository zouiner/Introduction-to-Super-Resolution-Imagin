#packages
from tkinter import N
import numpy as np
from skimage import io
from skimage.transform import rescale, resize_local_mean
from skimage.filters import correlate_sparse
from matplotlib import pyplot as plt