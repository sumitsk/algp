import scipy.io
import ipdb
from models import GPR
import pandas as pd
import numpy as np
from utils import zero_mean_unit_variance

# Intel Berkeley dataset
# mat = scipy.io.loadmat('data/intel_data.mat')
# x = mat['Xs']
# y = mat['Fs'].squeeze()
# y = zero_mean_unit_variance(y)

# Jura dataset
df = pd.read_pickle('data/jura_train_dataframe.pkl')
x = df[['xloc', 'yloc']].values
y = df[['Pb']].values.squeeze()
y = zero_mean_unit_variance(y)

# Gilgai dataset
# df = pd.read_csv('data/gilgai.csv')
# x = df[['x (m)']].values
# y = df[['pH (30-40 cm)']].values.squeeze()
# y = zero_mean_unit_variance(y)

nsims = 5
all_rho = []
for i in range(nsims):
	gp = GPR(lr=.1, max_iterations=200)
	gp.fit(x, y)

	params = dict(gp.model.named_parameters())
	ss = np.exp(params['kernel_covar_module.log_outputscale'].item())
	sn = np.exp(params['likelihood.log_noise'].item())
	rho = ss**2/sn**2
	all_rho.append(rho)

ipdb.set_trace()


# signal-to-noise ratio

# ===========================================
# Intel Berkeley Dataset
# [4.84, 5.03, 4.79, 4.64, 5.55]

# normalized 
# [4.73, 5.16, 5.24, 5.18, 5.13]

# ============================================
# Jura dataset
# normalized
# Ni - [92.77, 94.45, 91.15, 93.68, 92.88] 
# Cd - [5.02, 5.09, 5.07, 5.20, 4.78] 
# Zn - [25.01, 24.55, 24.84, 25.23, 24.67]
# Co - [28.03, 27.34, 27.62, 27.60, 28.09]
# Cr - [11.91, 11.89, 12.48, 6.33, 6.2]
# Pb - [52.49, 50.23, 47.87, 49.68, 50.02]
# Cu - [3.12, 3.18, 3.17, 3.15, 3.13]

# =============================================
# Sorghum July dataset
# plant_width - [2.04, 2.05, 1.24, 7.27, 1.68]
# plant_count - [0.22, 0.31, 0.23, 0.32, 0.29]
# plant_height - [10.65, 11.70, 15.19, 18.34, 12.94]

# normalized zero mean unit variance
# plant_width - [2.26, 0.09, 0.13, 0.51, 0.13]
# plant_count - [0.14, 0.20, 0.26, 0.23, 0.09]
# plant_height - [4.17, 0.30, 0.45, 0.51, 0.52]
