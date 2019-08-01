import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from chi_squared import chi_squared
import xgboost as xgb
class XGBJungle():
    #CSV FILES WITH COLUMNS ORDERED AS: PARAMETERS(cosmo + hod + radius) FOLLOWED BY LABEL, ERROR ON LABEL, BIN NUMBER
    def __init__(self, training_data_path, test_data_path, num_params=16, jungle_params=None):
        if training_data_path:
            train_data = np.genfromtxt(training_data_path, delimiter=',')
            train_data = train_data[train_data[:,-1].argsort()]
            self.x_train, self.y_train = train_data[:,:num_params], train_data[:,num_params]
            self.train_error, self.train_bin = train_data[:,num_params+1], train_data[:,num_params+2].astype(int)
        if test_data_path:
            test_data = np.genfromtxt(test_data_path, delimiter=',')
            test_data = test_data[test_data[:,-1].argsort()]
            self.x_test, self.y_test = test_data[:,:num_params], test_data[:,num_params]
            self.test_error, self.test_bin = test_data[:,num_params+1], test_data[:, num_params+2].astype(int)
        
        self.num_params=num_params
        self.bins=9
        if training_data_path:
            self.bin_size = self.x_train.shape[0]//self.bins
        self.bin_size_test = self.x_test.shape[0]//self.bins
        if not jungle_params:
            self.param_grid  = [{
                'subsample': 0.75, 
                'num_parallel_tree':10, 
                'num_boost_round':1, 
                'max_depth':15, 
                'objective':'reg:squarederror', 
                'eta':0.01} 
                for _ in range(9) ]
        else:
            self.param_grid=jungle_params
        print("FOREST PARAMETERS:", self.param_grid)
        
    def fit(self,save_path=None, num_round=2000, tolerance=3):
        if not 'x_train' in dir(self):
            print('No Training Data')
            return
        self.jungle=[]
        for i in range(self.bins):
            chunk = i*self.bin_size
            dtrain = xgb.DMatrix(self.x_train[chunk:chunk+self.bin_size], label=self.y_train[chunk:chunk+self.bin_size])
            
            chunk_test = i*self.bin_size_test
            dtest = xgb.DMatrix(self.x_test[chunk_test:chunk_test+self.bin_size_test], label=self.y_test[chunk_test:chunk_test+self.bin_size_test])
            print('Fitting Bin', i)
            bst = xgb.train(self.param_grid[i], dtrain, num_round, evals=[(dtrain, 'train'),(dtest, 'eval')], early_stopping_rounds=tolerance)
            self.jungle.append(bst)
        #SAVE NEW MODEL  
        if save_path:      
            joblib.dump(self.jungle, save_path)
    
    def load_model(self, load_path):
        #LOAD PRETRAINED FOREST
        self.jungle = joblib.load(load_path)
    
    def get_feature_importances(self):
        FI = [self.jungle[i].get_fscore() for i in range(self.bins)]
        return FI
    
    def score(self):
        print("\nTEST CHI^2")
        pred = np.zeros(self.y_test.shape)
        for i in range(self.bins):
            chunk_test = i*self.bin_size_test
            dtest = xgb.DMatrix(self.x_test[chunk_test:chunk_test+self.bin_size_test], label=self.y_test[chunk_test:chunk_test+self.bin_size_test])
            predictions = self.jungle[i].predict(dtest).tolist()
            for j in range(chunk_test, chunk_test + self.bin_size_test):
                pred[j] = predictions[j-chunk_test]
        print(chi_squared(pred,self.y_test,self.test_error))
    
        print("\nTEST FRACTIONAL ERROR")
        fractional_error = np.zeros((self.bins,1))
        #frac_err = np.abs(pred-self.y_test)/self.y_test
        frac_err = np.abs(pred-self.y_test)/self.test_error
        for bin, err in zip(self.test_bin, frac_err):
            fractional_error[bin] += err
        fractional_error = fractional_error / (len(self.test_bin)/self.bins)  #average over number of elements per bin
        print(fractional_error)

        print("\nPREDICTIONS")
        print(pred)

    
    def infer(self, infer_parameters):
        #pass path to parameters or parameters as a list, bin as last parameter
        if infer_parameters is str:
            parameters = np.genfromtxt(infer_parameters, delimiter=',')
        else:
            parameters = infer_parameters
        idx = parameters[-1].item()
        X = xgb.DMatrix(parameters[:-1])
      
        prediction = self.jungle[idx].predict(X)
        return prediction
