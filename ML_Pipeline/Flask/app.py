from flask import Flask, render_template, request
import re
import pandas as pd
import copy
import pickle
import joblib

deploy = pickle.load(open('mod.pkl','rb'))
impute = joblib.load('medianimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')
mostfrequent = joblib.load('mostfrequent')

# connecting to sql by creating sqlachemy engine
from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",#user
                               pw = "user1", # passwrd
                               db = "knn"))

def multinomial_reg(data_new):
    clean1 = pd.DataFrame(impute.transform(data_new), columns = data_new.select_dtypes(exclude = ['object']).columns)
    clean1[['Age']] = winsor.transform(clean1[['Age']])
    clean2 = pd.DataFrame(minmax.transform(clean1))
    clean3 = pd.DataFrame(mostfrequent.transform(data_new), columns=data_new.select_dtypes(include=['object']).columns)
    clean4 = pd.DataFrame(encoding.transform(clean3).todense())
    clean_data = pd.concat([clean2, clean4], axis = 1, ignore_index = True)
    prediction = pd.DataFrame(deploy.predict(clean_data), columns = ['Output'])
    final_data = pd.concat([prediction, data_new], axis = 1)
    return(final_data)
    
            
#define flask
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data_new = pd.read_excel(f)
       
        final_data = multinomial_reg(data_new)

        final_data.to_sql('credit_test', con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        
        
       
        return render_template("new.html", Y = final_data.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True)
