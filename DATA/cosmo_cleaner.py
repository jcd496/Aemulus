import numpy as np
from sklearn.preprocessing import StandardScaler
hod_params = 8
cosmo_params = 7
rbin= 9

#radius + wp + error + bin_num= +4 params for total of 19 columns
#40 cosmo x 2000 hod x 9 radius buckets = 720,000 rows of data
hod_data = np.genfromtxt("HOD_parameters_5000.dat", delimiter=" ", dtype=float)
cosmo_data = np.genfromtxt("cosmological_parameters_full.dat", delimiter="", dtype=float)
all_data_training=np.empty((720000,19))

for cosmo in range(40): #40

	for hod in range(2000): #2000
		wp_path = "training/wp_covar_results/wp_covar_cosmo_{}_HOD_{}_test_0.dat".format(cosmo, hod)
		wp_covar = np.genfromtxt(wp_path, delimiter="", dtype=float)
			
		 
		
		for rad in range(9):#9
			wp = wp_covar[rad,1]
			radius = wp_covar[rad,0]
			error = wp_covar[rad,2]
			full_data = np.append(cosmo_data[cosmo],hod_data[hod])
			full_data = np.append(full_data, radius)
			full_data = np.append(full_data, wp)
			full_data = np.append(full_data, error)	
            full_data = np.append(full_data, rad)  #bin number
			
			all_data_training[cosmo*2000*rbin + hod*rbin+rad]=full_data

np.random.shuffle(all_data_training)

x = all_data_training[:,:16]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
all_data_training = np.append(x_scaled, all_data_training[:,16:], 1)
np.savetxt('cosmo_training_data.csv', all_data_training[:540000,:], fmt='%f', delimiter=',')
np.savetxt('cosmo_test_data.csv',all_data_training[540000:,:],fmt='%f',delimiter=',')

