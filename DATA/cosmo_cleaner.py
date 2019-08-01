import numpy as np 
from sklearn.preprocessing import StandardScaler

cosmo_params=7
hod_params=8
rbin=9
SCALE=False
#radius + wp + error= + 3 params for total of 18 columns
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
			full_data = np.append(full_data, rad) #bin number	
			
				#complete_training_data[2000*9*cosmo+9*hod+rad]=full_data
			all_data_training[cosmo*2000*rbin + hod*rbin+rad]=full_data

np.random.shuffle(all_data_training)
if(SCALE):
	x = all_data_training[:,:16]
	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)
	all_data_training = np.append(x_scaled, all_data_training[:,16:],1)

	np.savetxt('scaled_training_data.csv', all_data_training, fmt='%f', delimiter=',')



	#break here

	all_data_test=np.empty((9,19))

	cosmo = 0
	hod = 39
	wp_path = "test/wp_covar_output/wp_covar_cosmo_{}_Box_2_HOD_{}_test_0.dat".format(cosmo, hod)
	wp_covar = np.genfromtxt(wp_path, delimiter="", dtype=float)
		
	 

	for rad in range(9):#9
		wp = wp_covar[rad,1]
		radius = wp_covar[rad,0]
		error = wp_covar[rad,2]
		full_data = np.append(cosmo_data[cosmo],hod_data[hod])
		full_data = np.append(full_data, radius)
		full_data = np.append(full_data, wp)
		full_data = np.append(full_data, error)
		full_data = np.append(full_data, rad) #bin number	
		
			#complete_test_data[2000*9*cosmo+9*hod+rad]=full_data
		all_data_test[rad]=full_data
	x = all_data_test[:,:16]
	x_scaled = scaler.transform(x)	
	test = np.append(x_scaled, all_data_test[:,16:],1)
	np.savetxt('scaled_validation_data.csv', test, fmt='%f', delimiter=',')
else:
	np.savetxt('entire_training_data.csv', all_data_training, fmt='%f', delimiter=',')



