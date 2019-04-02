import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from chi_squared import chi_squared

class Regressor():
    #CSV FILES WITH COLUMNS ORDERED AS: PARAMETERS FOLLOWED BY LABEL AND ENDING WITH ERROR ON LABEL
    def __init__(self, training_data_path, test_data_path, num_params=16, forest_params=None):
        if training_data_path:
            train_data = np.genfromtxt(training_data_path, delimiter=',')
            self.x_train, self.y_train = train_data[:,:num_params], train_data[:,num_params]
            self.train_error = train_data[:,-1]
        if test_data_path:
            test_data = np.genfromtxt(test_data_path, delimiter=',')
            self.x_test, self.y_test = test_data[:,:num_params], test_data[:,num_params]
            self.test_error = test_data[:,-1]

        if not forest_params:
            self.param_grid = {
                "n_estimators": 10,
                "criterion": 'mse',
                "max_depth": 16,
                "max_features": 16,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "bootstrap": True
            }
        else:
            self.param_grid=forest_params
        print("FOREST PARAMETERS:", self.param_grid)
        self.forest = RandomForestRegressor(**self.param_grid)
        
        
        def grid_search(self):
            grid = GridSearchCV(estimator=self.forest, param_grid=self.param_grid, n_jobs=1)
            grid.fit(self.x_train,self.y_train)
            print(grid.best_score_) 
            print(grid.best_params_) 
            return grid.best_params_

    #ADD ERROR HANDLING try fit, catch if unfit
    def fit(self,save_path=None):
        self.forest.fit(self.x_train,self.y_train)

        #SAVE NEW MODEL  
        if save_path:      
            joblib.dump(forest, save_path)
    
    def load_model(self, load_path):
        #LOAD PRETRAINED FOREST
        forest = joblib.load(load_path)

    def get_feature_importances(self):
        return self.forest.feature_importances_

    def score(self):
        print("\nR^2 SCORE")
        print(self.forest.score(self.x_test,self.y_test))

        print("\nTRAIN CHI^2")
        train_pred = self.forest.predict(self.x_train)
        print(chi_squared(train_pred,self.y_train,self.train_error))
    
        print("\nTEST CHI^2")
        pred = self.forest.predict(self.x_test)
        print(chi_squared(pred,self.y_test,self.test_error))

        print("\nTEST FRACTIONAL ERROR") 
        print(np.average(np.abs(pred-self.y_test)/self.y_test * 100))

        print("\nPREDICTIONS")
        print(pred)
    def infer(self, infer_parameters):
        #pass path to parameters or parameters as a list
        if infer_parameters is str:
            parameters = np.genfromtxt(infer_parameters, delimiter=',')
        else:
            parameters = infer_parameters
        return self.forest.predict(parameters)

