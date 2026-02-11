import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


class EnsembleClassifier():
    def __init__(self, model_name, scaler_name=None, random_state=42):
        self.random_state = random_state
        self.initialize_pipe(model_name, scaler_name)
    
    def model_zoo(self):
        model_zoo = [
                     # Bagging
                     "DecisionTree", 
                     # Ensemble Bagging+random
                     "RandomForest", 
                     # Boosting
                     "GradientBoosting",
                     "AdaBoost",
                    ]
    
        print( "Model zoo:", model_zoo, "\n" )
    
    def get_model_name(self):
        return self.model_name
    
    def get_pipeline(self):
        return self.pipeline
    
    def get_model(self):
        return self.model_grid
        
    def get_grid_params(self):
        if "decisiontree" == self.model_name.lower():
            parameters = {'classifier__max_features': [None, 'sqrt'],
                  'classifier__max_depth': np.append(None, np.arange(10, 80, 10) ), 
                  'classifier__min_samples_split': np.arange(3, 7), 
                  'classifier__min_samples_leaf': np.arange(3, 7), 
                 }
                 
        elif "randomforest" == self.model_name.lower():
            parameters = {'classifier__n_estimators': [200], #np.arange(100, 1100, 100),
                  'classifier__bootstrap': [True], #[True, False],
                  # Tree parameters
                  'classifier__max_features': [None], #[None, 'sqrt'],
                  'classifier__max_depth': [20], #np.append(None, np.arange(10, 80, 10) ),
                  'classifier__min_samples_split': [6], #np.arange(3, 7),
                  'classifier__min_samples_leaf': [3], #np.arange(3, 7),
                 }
                 
        elif "gradientboosting" == self.model_name.lower():
            parameters = {'classifier__n_estimators': np.arange(100, 1100, 100),
                  'classifier__subsample': [1.0],
                  'classifier__max_features': [None, 'sqrt'],
                  'classifier__max_depth': np.append(None, np.arange(10, 80, 10) ),
                  'classifier__min_samples_split': np.arange(3, 7),
                  'classifier__min_samples_leaf': np.arange(3, 7),
                 }
                 
        elif "adaboost" == self.model_name.lower():
            parameters = {'classifier__n_estimators': np.arange(100, 1100, 100),
                  'classifier__learning_rate': np.arange(0.1, 1.1, 0.1),
                 }
        
        else:
            parameters = None
        
        return parameters
    
   
    def initialize_grid(self, parameters=None, n_splits=5, n_jobs=5):
    
        parameters = self.get_grid_params() if parameters is None else parameters
    
        self.model_grid = GridSearchCV(estimator=self.pipeline,
                             param_grid=parameters,
                             scoring='balanced_accuracy',
                             n_jobs=n_jobs,
                             refit=True,
                             cv=StratifiedKFold(n_splits=n_splits),  # << Use time series
                             verbose=0,
                      )
    
    def initialize_pipe(self, model_name, scaler_name):
        self.initialize_scaler(scaler_name)
        self.initialize_model(model_name)
        
        steps = []
        if self.scaler is not None:
            steps.append(('scaler', self.scaler))
        
        if self.model is not None:
            steps.append(('classifier', self.model))
        
        self.pipeline = Pipeline(
                        steps = steps,
                        #memory=memory,
                    )
    
    def initialize_scaler(self, scaler_name):
        if scaler_name is None:
            self.scaler = None        
        elif scaler_name.lower() == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = None
    
    def initialize_model(self, model_name):
        if "decisiontree" == model_name.lower():
            self.model = DecisionTreeClassifier(random_state=self.random_state)
            self.model_name = model_name
            
        elif "randomforest" == model_name.lower():
            self.model = RandomForestClassifier(random_state=self.random_state)
            self.model_name = model_name
            
        elif "gradientboosting" == model_name.lower():
            self.model = GradientBoostingClassifier(random_state=self.random_state)
            self.model_name = model_name
            
        elif "adaboost" == model_name.lower():
            self.model = AdaBoostClassifier(random_state=self.random_state)
            self.model_name = model_name
            
        else:
            self.model = None
            self.model_name = None
            
    
