import numpy as np 
hod_params = 8
cosmo_params = 7
rbin= 9

#radius + wp + error= + 3 params for total of 18 columns
#40 cosmo x 2000 hod x 9 radius buckets = 720,000 rows of data
hod_data = np.genfromtxt("HOD_parameters_5000.dat", delimiter=" ", dtype=float)
cosmo_data = np.genfromtxt("cosmological_parameters_full.dat", delimiter="", dtype=float)
all_data_training=np.empty((701649,18))

for cosmo in range(39): #40

	for hod in range(1999): #2000
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
			
				#complete_training_data[2000*9*cosmo+9*hod+rad]=full_data
			all_data_training[cosmo*1999*rbin + hod*rbin+rad]=full_data

np.random.shuffle(all_data_training)
np.savetxt('cosmo_training_data.csv', all_data_training[:540000,:], fmt='%f', delimiter=',')
np.savetxt('cosmo_test_data.csv',all_data_training[540000:,:],fmt='%f',delimiter=',')


cosmo = 39
hod = 1999
wp_path = "training/wp_covar_results/wp_covar_cosmo_{}_HOD_{}_test_0.dat".format(cosmo, hod)
wp_covar = np.genfromtxt(wp_path, delimiter="", dtype=float)
	
cosmo_39_hod_1999 = np.empty((9,18))		 
		
for rad in range(9):#9
	wp = wp_covar[rad,1]
	radius = wp_covar[rad,0]
	error = wp_covar[rad,2]
	plot_data = np.append(cosmo_data[cosmo],hod_data[hod])
	plot_data = np.append(plot_data, radius)
	plot_data = np.append(plot_data, wp)
	plot_data = np.append(plot_data, error)	
			
		#complete_training_data[2000*9*cosmo+9*hod+rad]=full_data
	cosmo_39_hod_1999[rad]=plot_data
np.savetxt('cosmo_39_hod_1999.csv', cosmo_39_hod_1999, fmt='%f', delimiter=',')

"""print("total", hod_data_training.shape)
print("training",hod_data_training[:14000,:].shape)
print("test",hod_data_training[14000:,:].shape)"""
