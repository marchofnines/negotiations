import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import dump, load

from helpers.my_imports import * 
from helpers import tools
from helpers import transformers
from helpers import preprocessing
from IPython.core.display import HTML
from pandas.io.formats import style


#Global Variable for Random State
rs=42 #random_state

df = pd.read_csv('saved_dfs/preprocessed_negotiations_df.csv')
X,y=df.drop(columns=['decision', 'billed_amount', 'negotiation_amount', 'offer', 'counter_offer']), df.decision

X_gbc, y_gbc = X.copy(deep=True), y.copy(deep=True) 
X_gbc[X_gbc.select_dtypes(exclude=['number']).columns] = X_gbc.select_dtypes(exclude=['number']).astype('str')
y_gbc = y_gbc.map({'Accepted':1, 'Rejected':0})
X_train_gbc, X_test_gbc, y_train_gbc, y_test_gbc = train_test_split(X_gbc,y_gbc, stratify=y_gbc, test_size=0.25, random_state=42)
X_train_gbc.shape, X_test_gbc.shape, y_train_gbc.shape, y_test_gbc.shape


model = load('models/hyperparam_tuning/ht_ens_set8.joblib')

#Calculate predicted train and test target
#y_pred_train = model.predict(X_train_gbc)
#y_pred_test = model.predict(X_test_gbc)

# Calculating F1 score
#f1w_train_gbc = f1_score(y_train_gbc, y_pred_train, average='weighted')
#f1w_test_gbc = f1_score(y_test_gbc, y_pred_test, average='weighted')
#print(f1w_train_gbc)
#print(f1w_test_gbc)

#data validation
class input_vars(BaseModel): 
    claim_type: str
    NSA_NNSA: str
    split_claim: str
    negotiation_type: str
    in_response_to: str
    level: str
    facility: str
    carrier: str
    group_number: str
    plan_funding: str
    TPA: str
    TPA_rep: str
    service_days: float	
    decision_days: float
    offer_days: float	
    counter_offer_days: float	
    YOB: int    
    neg_to_billed: float	
    offer_to_neg: float
    offer_to_counter_offer:float 
    
app = FastAPI()
    
@app.get("/")
def read_root(): 
    return {"Hello": "World"}


@app.post("/predict_negotiation_decision/")
def make_preds(independent_variables: input_vars):
    prediction = model.predict(pd.DataFrame(independent_variables.model_dump(), index=[0] ))
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)