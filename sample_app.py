import streamlit as st
import requests
import json

#st.header("Claim Negotiation Decision Predictor")
st.header("Please enter the Negotiation details below:")

payload_dict={}
payload_dict['claim_type'] = st.selectbox("Claim Type", ['UB', 'HCFA'])
payload_dict['NSA_NNSA'] = 'NSA' if st.checkbox("NSA?") else 'NNSA'
payload_dict['split_claim'] = 'Yes' if st.checkbox("Split Claim?") else 'No'
payload_dict['negotiation_type'] = st.selectbox("Negotiation Type", ['NNSA Negotiation', 'Open Negotiation', 'Prepayment Negotiation'])
payload_dict['in_response_to'] = st.selectbox("In Response To", ['Open Negotiation Notice', 'Insurance Initiated', 'Negotiation Letter', 'Verbal Request', 'Reconsideration Letter', 'Corrected Claim'])
payload_dict['level'] = st.selectbox("Level", ['Level 4', 'Level 3', 'Level 2', 'Level 1'])
payload_dict['facility'] = st.selectbox("Facility", ["Benbrook", "Denton", "Cedar Hill", "Desoto", "Garland", "Frisco", "Matlock"])
payload_dict['carrier'] = st.text_input(label='Carrier')
payload_dict['group_number'] = st.text_input(label='Group Number')
payload_dict['plan_funding'] = 'FULLY' if st.checkbox("Fully Funded?") else 'SELF'
payload_dict['TPA'] = st.text_input(label='TPA')
payload_dict['TPA_rep'] = st.text_input(label='TPA Rep')
payload_dict['service_days'] = st.number_input(label='Number of Days Between Service Date and Deadline', step=1)
payload_dict['decision_days'] = st.number_input(label='Number of Days Between Decision Date and Deadline', step=1)
payload_dict['offer_days'] = st.number_input(label='Number of Days Between Offer Date and Deadline', step=1)
payload_dict['counter_offer_days'] = st.number_input(label='Number of Days Between Counter Offer Date and Deadline', step=1)
payload_dict['YOB'] = st.number_input(label='Year of Birth', step=1)
payload_dict['neg_to_billed'] = st.number_input(label='Ratio of Negotiated Amount To Billed Amount', format="%f", step=0.0001)
payload_dict['offer_to_neg'] = st.number_input(label='Ratio of Offer to Negotiated Amount', format="%f", step=0.0001)
payload_dict['offer_to_counter_offer'] = st.number_input(label='Ratio of Offer to Counter Offer', format="%f", step=0.0001)


url="http://127.0.0.1:8000/predict_negotiation_decision/"

response = requests.post(url=url, json=payload_dict)



if response.status_code !=200:
    st.error(f"Error:  Status Code {response.status_code}.  Input {payload_dict}")
else:
    st.success("Processed Successfully")
    prediction = json.loads(response.text)['prediction']
    st.write("## Prediction: Accepted" if prediction ==1 else "## Prediction: Rejected")
    