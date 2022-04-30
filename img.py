import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./datasets/ISCX_Botnet.csv")

from matplotlib import pyplot
df.hist()
pyplot.show()