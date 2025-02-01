import numpy as np

from utils.utils import config

atom_R = np.random.randn(config.dimension, config.dimension)
atom_e = np.random.randn(config.num_nodes)
coef = 2


tmp_3 = coef * np.identity(config.dimension) * np.sum(atom_e**2)

tmp4 = coef * np.eye(config.dimension) * np.sum(atom_e**2)
