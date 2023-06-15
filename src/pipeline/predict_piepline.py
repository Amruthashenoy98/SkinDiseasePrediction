import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        erythema, scaling, definite_borders, itching,
       koebner_phenomenon, follicular_papules,
       knee_and_elbow_involvement, scalp_involvement, family_history,
       eosinophils_in_the_infiltrate, PNL_infiltrate,
       fibrosis_of_the_papillary_dermis, exocytosis, acanthosis,
       hyperkeratosis, parakeratosis, elongation_of_the_rete_ridges,
       spongiform_pustule, munro_microabcess,
       disappearance_of_the_granular_layer, spongiosis,
       inflammatory_monoluclear_inflitrate, Age):

        self.erythema=erythema, 
        self.scaling=scaling,
        self.definite_borders=definite_borders, 
        self.itching=itching,
        self.koebner_phenomenon=koebner_phenomenon, 
        self.follicular_papules=follicular_papules,
        self.knee_and_elbow_involvement=knee_and_elbow_involvement, 
        self.scalp_involvement=scalp_involvement, 
        self.family_history=family_history,
        self.eosinophils_in_the_infiltrate=eosinophils_in_the_infiltrate, 
        self.PNL_infiltrate=PNL_infiltrate,
        self.fibrosis_of_the_papillary_dermis=fibrosis_of_the_papillary_dermis, 
        self.exocytosis=exocytosis, 
        self.acanthosis=acanthosis,
        self.hyperkeratosis=hyperkeratosis, 
        self.parakeratosis=parakeratosis,
        self.elongation_of_the_rete_ridges=elongation_of_the_rete_ridges,
        self.spongiform_pustule=spongiform_pustule, 
        self.munro_microabcess=munro_microabcess,
        self.disappearance_of_the_granular_layer=disappearance_of_the_granular_layer, 
        self.spongiosis=spongiosis,
        self.inflammatory_monoluclear_inflitrate=inflammatory_monoluclear_inflitrate, 
        self.Age=Age
        
        print(self.erythema, self.scaling, self.definite_borders, self.itching,
                 self.koebner_phenomenon, self.follicular_papules,
                 self.knee_and_elbow_involvement, self.scalp_involvement, self.family_history,
                 self.eosinophils_in_the_infiltrate, self.PNL_infiltrate,
                 self.fibrosis_of_the_papillary_dermis, self.exocytosis, self.acanthosis,
                 self.hyperkeratosis, parakeratosis, self.elongation_of_the_rete_ridges,
                 self.spongiform_pustule, self.munro_microabcess,
                 self.disappearance_of_the_granular_layer, self.spongiosis,
                 self.inflammatory_monoluclear_inflitrate, self.Age)
       

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
            'erythema' :  [int(self.erythema[0])], 
            'scaling' : [int(self.scaling[0])],
            'definite_borders' : [int(self.definite_borders[0])], 
            'itching' : [int(self.itching[0])],
            'koebner_phenomenon': [int(self.koebner_phenomenon[0])], 
            'follicular_papules': [int(self.follicular_papules[0])],
            'knee_and_elbow_involvement':[int(self.knee_and_elbow_involvement[0])], 
            'scalp_involvement':[int(self.scalp_involvement[0])], 
            'family_history':[int(self.family_history[0])],
            'eosinophils_in_the_infiltrate':[int(self.eosinophils_in_the_infiltrate[0])], 
            'PNL_infiltrate':[int(self.PNL_infiltrate[0])],
            'fibrosis_of_the_papillary_dermis':[int(self.fibrosis_of_the_papillary_dermis[0])], 
            'exocytosis':[int(self.exocytosis[0])], 
            'acanthosis':[int(self.acanthosis[0])],
            'hyperkeratosis':[int(self.hyperkeratosis[0])], 
            'parakeratosis':[int(self.parakeratosis[0])],
            'elongation_of_the_rete_ridges':[int(self.elongation_of_the_rete_ridges[0])],
            'spongiform_pustule':[int(self.spongiform_pustule[0])], 
            'munro_microabcess':[int(self.munro_microabcess[0])],
            'disappearance_of_the_granular_layer':[int(self.disappearance_of_the_granular_layer[0])], 
            'spongiosis':[int(self.spongiosis[0])],
            'inflammatory_monoluclear_inflitrate':[int(self.inflammatory_monoluclear_inflitrate[0])], 
            'Age':[self.Age]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)