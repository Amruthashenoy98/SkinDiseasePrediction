from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_piepline import CustomData,PredictPipeline

application = Flask(__name__) #entrypoint to execute

app = application

#create route for home page

@app.route('/')
def index():
    return render_template('index.html') #searches for template folder

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html') #home.html will contains the data fields that user gives as input to make prediction
    else:
         data=CustomData(
            erythema =  request.form.get('erythema'), 
            scaling = request.form.get('scaling'),
            definite_borders = request.form.get('definite_borders'), 
            itching =request.form.get('itching'),
            koebner_phenomenon= request.form.get('koebner_phenomenon'), 
            follicular_papules= request.form.get('follicular_papules'),
            knee_and_elbow_involvement=request.form.get('knee_and_elbow_involvement'), 
            scalp_involvement=request.form.get('scalp_involvement'), 
            family_history=request.form.get('family_history'),
            eosinophils_in_the_infiltrate=request.form.get('eosinophils_in_the_infiltrate'), 
            PNL_infiltrate=request.form.get('PNL_infiltrate'),
            fibrosis_of_the_papillary_dermis=request.form.get('fibrosis_of_the_papillary_dermis'), 
            exocytosis=request.form.get('exocytosis'), 
            acanthosis=request.form.get('acanthosis'),
            hyperkeratosis=request.form.get('hyperkeratosis'), 
            parakeratosis=request.form.get('parakeratosis'),
            elongation_of_the_rete_ridges=request.form.get('elongation_of_the_rete_ridges'),
            spongiform_pustule=request.form.get('spongiform_pustule'), 
            munro_microabcess=request.form.get('munro_microabcess'),
            disappearance_of_the_granular_layer=request.form.get('disappearance_of_the_granular_layer'), 
            spongiosis=request.form.get('spongiosis'),
            inflammatory_monoluclear_inflitrate=request.form.get('inflammatory_monoluclear_inflitrate'), 
            Age=request.form.get('Age')

        )
         pred_df = data.get_data_as_data_frame()
         print(pred_df)
         print("Before Prediction")

         predict_pipeline=PredictPipeline()
         print("Mid Prediction")
         results=predict_pipeline.predict(pred_df)
         prediction_mapping = {
            0: "psoriasis",
            1: "seborrheic dermatitis",
            2: "lichen planus",
            3: "pityriasis rosea",
            4: "chronic dermatitis",
            5: "pityriasis rubra pilaris"
            }
         print("after Prediction")
         return render_template('home.html',results=prediction_mapping.get(results[0], 'Unknown'))
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
    