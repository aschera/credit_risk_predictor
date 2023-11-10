import streamlit as st
import pandas as pd
import plotly.express as px
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

relative_path = os.path.join(
    current_directory,
    "..",
    "static",
    "final_dataset.csv"
)

st.header("Place for Interactive data page")

@st.cache_data()
def load_data():
    df = pd.read_csv(relative_path)
    return df

df = load_data()

st.markdown("""
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 250px;
            font-size: 1rem;
            color: black; 
            background-color: #fff87f;
        }
    </style>
    """, unsafe_allow_html=True)

#Function to map columns
def map_column(column, labels):
    return df[column].map(labels)

#Label dictionaries
action_taken_labels = {
    1: 'Accepted',
    3: 'Rejected'
}
loan_type_labels = {
    1: 'Conventional',
    2: 'FHA',
    3: 'VA',
    4: 'RHS or FSA'
}

lien_status_labels = {
    1: 'Secured by a first lien',
    2: 'Secured by a subordinate lien'
}

line_of_credit_labels = {
    1: 'Open-end line of credit',
    2: 'Not an open-end line of credit'
}
combined_loanvalue_labels = {
    1:'75.0-80.0',
    2:'70.0-75.0',
    3:'95.0-100.0',
    4:'65.0-70.0',
    5:'90.0-95.0',
    6:'85.0-90.0',
    7:'55.0-60.0',
    8:'60.0-65.0',
    9:'50.0-55.0',
    10:'80.0-85.0',
    11:'45.0-50.0',
    12:'40.0-45.0',
    13:'35.0-40.0',
    14:'30.0-35.0',
    15:'25.0-30.0',
    16:'100.0-120.0',
    17:'0.0-25.0'
}
interest_rate_labels = {
    1:'0.0-1.0',
    2:'1.0-2.0',
    3:'2.0-3.0',
    4:'3.0-3.5',
    5:'3.5-4.0',
    6:'4.0-4.5',
    7:'4.5-5.0',
    8:'5.0-6.0',
    9:'6.0-7.0',
    10:'7.0-8.0',
    11:'8.0-9.0'
}
total_loan_costs_labels = {
    1:'2500-3000',
    2:'3000-3500',
    3:'3500-4000',
    4:'500-1000',
    5:'2000-2500',
    6:'4000-4500',
    7:'0-500',
    8:'4500-5000',
    9:'1500-2000',
    10:'5000-5500',
    11:'1000-1500',
    12:'5500-6000',
    13:'6000-6500',
    14:'6500-7000',
    15:'7000-7500',
    16:'7500-8000',
    17:'8000-8500',
    18:'8500-9000'
}
origination_charges_labels = {
    1:'0-500',
    2:'1000-1500',
    3:'500-1000',
    4:'1500-2000',
    5:'2000-2500',
    6:'2500-3000',
    7:'3000-3500',
    8:'3500-4000'
}
loan_term_labels = {
    1:'2-6 years',
    2:'7-11 years',
    3:'12-16 years',
    4:'17-21 years',
    5:'22-26 years',
    6:'27-31 years',
    7:'37-41 years'
}
negative_amortization_labels ={
    1: "Negative amortization",
    2: "No negative amortization"
}
interest_only_payment_labels = {
    1: "Interest-only payments",
    2: "No interest-only payments"
}
balloon_payment_labels = {
    1: "Balloon payment",
    2: "No baloon payment"
}
other_nonamortizing_features_labels = {
    1: "Other non-fully amortizing features",
    2: "No other non-fully amortizing features"
}
occupancy_type_labels = {
    1: "Principal residence",
    2: "Second residence",
    3: "Investment property"
}
manufactured_home_secured_property_type_labels = {
    1: "Manufactured home and land",
    2: "Manufactured home and not land",
    3: "Not Applicable"
}
manufactured_home_land_property_interest_labels = {
    1: "Direct ownership",
    2: "Indirect ownership",
    3: "Paid leasehold",
    4: "Unpaid leasehold",
    5: "Not applicable"
}
total_units_labels = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5-24"
}
debt_to_income_ratio_labels = {
    1:"<20%",
    2:'20%-<30%',
    3:'30%-<36%',
    4:'36%-50%',
    5:'50%-60%',
    6:'>60%'

}
applicant_credit_score_type_labels = {
    1: "Equifax Beacon 5.0",
    2: "Experian Fair Isaac",
    3: "FICO Risk Score Classic 04",
    4: "FICO Risk Score Classic 98",
    5: "VantageScore 2.0",
    6: "VantageScore 3.0",
    7: "More than one credit scoring model",
    8: "Other credit scoring model",
    9: "Not applicable"
}
co_applicant_credit_score_type_labels = {
    1: "Equifax Beacon 5.0",
    2: "Experian Fair Isaac",
    3: "FICO Risk Score Classic 04",
    4: "FICO Risk Score Classic 98",
    5: "VantageScore 2.0",
    6: "VantageScore 3.0",
    7: "More than one credit scoring model",
    8: "Other credit scoring model",
    9: "Not applicable",
    10: "No co-applicant"
}
applicant_ethnicity_labels = {
    2: "Not Hispanic or Latino",
    1: "Hispanic or Latino"
}
co_applicant_ethnicity_labels = {
    2: "Not Hispanic or Latino",
    1: "Hispanic or Latino"
}
applicant_race_1_labels = {
    1:"American Indian or Alaska Native",
    2: "Asian",
    3: "Black or African American",
    4: "Native Hawaiian or Other Pacific Islander",
    5: "White",
    6: "Information not provided by applicant in mail, internet, or telephone application",
    7: "Not applicable"
}
applicant_race_2_labels = {
    1:"American Indian or Alaska Native",
    2: "Asian",
    3: "Black or African American",
    4: "Native Hawaiian or Other Pacific Islander",
    5: "White",
    6: "Information not provided by applicant in mail, internet, or telephone application",
    7: "Not applicable"
}
co_applicant_race_1_labels = {
    1:"American Indian or Alaska Native",
    2: "Asian",
    3: "Black or African American",
    4: "Native Hawaiian or Other Pacific Islander",
    5: "White",
    6: "Information not provided by applicant in mail, internet, or telephone application",
    7: "Not applicable",
    8: "No co-applicant"
}
co_applicant_race_2_labels = {
    1:"American Indian or Alaska Native",
    2: "Asian",
    3: "Black or African American",
    4: "Native Hawaiian or Other Pacific Islander",
    5: "White", 
}
applicant_sex_labels = {
    1: "Male",
    2: "Female", 
}
co_applicant_sex_labels = {
    1: "Male",
    2: "Female", 
}
applicant_age_labels = {
    0: "25-34",
    1:"35-44",
    2: "45-54",
    3: "55-64",
    4: "65-74",
    6: "<25",
    7: ">74"
}
co_applicant_age_labels = {
    0: "25-34",
    1:"35-44",
    2: "45-54",
    3: "55-64",
    4: "65-74",
    6: "<25",
    7: ">74"
}

df["action_taken"] = map_column('action_taken', action_taken_labels)
df['loan_type'] = map_column('loan_type', loan_type_labels)
df['lien_status'] = map_column('lien_status', lien_status_labels)
df['open_end_line_of_credit'] = map_column('open_end_line_of_credit', line_of_credit_labels)
df['combined_loan_to_value_ratio'] = map_column('combined_loan_to_value_ratio', combined_loanvalue_labels)
df['interest_rate'] = map_column('interest_rate', interest_rate_labels)
df['total_loan_costs'] = map_column('total_loan_costs', total_loan_costs_labels)
df['origination_charges'] = map_column('origination_charges', origination_charges_labels)
df['loan_term'] = map_column('loan_term', loan_term_labels)
df['negative_amortization'] = map_column('negative_amortization', negative_amortization_labels)
df['interest_only_payment'] = map_column('interest_only_payment', interest_only_payment_labels)
df['balloon_payment'] = map_column('balloon_payment', balloon_payment_labels)
df['other_nonamortizing_features'] = map_column('other_nonamortizing_features', other_nonamortizing_features_labels)
df['occupancy_type'] = map_column('occupancy_type', occupancy_type_labels)
df['manufactured_home_secured_property_type'] = map_column('manufactured_home_secured_property_type', manufactured_home_secured_property_type_labels)
df['manufactured_home_land_property_interest'] = map_column('manufactured_home_land_property_interest', manufactured_home_land_property_interest_labels)
df['total_units'] = map_column('total_units', total_units_labels)
df['debt_to_income_ratio'] = map_column('debt_to_income_ratio', debt_to_income_ratio_labels)
df['applicant_credit_score_type'] = map_column('applicant_credit_score_type', applicant_credit_score_type_labels)
df['co_applicant_credit_score_type']= map_column('co_applicant_credit_score_type', co_applicant_credit_score_type_labels)
df['applicant_ethnicity_1'] = map_column('applicant_ethnicity_1', applicant_ethnicity_labels)
df['co_applicant_ethnicity_1'] = map_column('co_applicant_ethnicity_1', co_applicant_ethnicity_labels)
df['applicant_race_1'] = map_column('applicant_race_1', applicant_race_1_labels)
df['applicant_race_2']= map_column('applicant_race_2', applicant_race_2_labels)
df['co_applicant_race_1']= map_column('co_applicant_race_1', co_applicant_race_1_labels)
df['co_applicant_race_2'] = map_column('co_applicant_race_2', co_applicant_race_2_labels)
df['applicant_sex'] = map_column('applicant_sex', applicant_sex_labels)
df['co_applicant_sex'] = map_column('co_applicant_sex', co_applicant_sex_labels)
df['applicant_age'] = map_column('applicant_age', applicant_age_labels)
df['co_applicant_age'] = map_column('co_applicant_age', co_applicant_age_labels)
df['income'] = df['income'] * 1000

display_df = df


with st.sidebar:
    st.header("Filters")
    selected_applicant_ethnicity = st.sidebar.multiselect(
        "Select applicant ethnicity",
        options = df['applicant_ethnicity_1'].unique(),
        default=df['applicant_ethnicity_1'].unique(),
    )
    selected_applicant_race = st.sidebar.multiselect(
        "Select applicant race",
        options=df['applicant_race_1'].unique(),
        default=df['applicant_race_1'].unique(),
    )
    selected_applicant_sex = st.sidebar.multiselect(
        "Select applicant sex",
        options=df['applicant_sex'].unique(),
        default=df['applicant_sex'].unique(),
    )
    selected_applicant_age = st.sidebar.multiselect(
        "Select applicant age range",
        options=df['applicant_age'].unique(),
        default=df['applicant_age'].unique(),
    )
    selected_co_applicant_ethnicity = st.sidebar.multiselect(
        "Select co-applicant ethnicity",
        options = df['co_applicant_ethnicity_1'].unique(),
        default=df['co_applicant_ethnicity_1'].unique(),
    )
    selected_co_applicant_race = st.sidebar.multiselect(
        "Select co-applicant race",
        options=df['co_applicant_race_1'].unique(),
        default=df['co_applicant_race_1'].unique(),
    )
    selected_co_applicant_sex = st.sidebar.multiselect(
        "Select co-applicant sex",
        options=df['co_applicant_sex'].unique(),
        default=df['co_applicant_sex'].unique(),
    )
    selected_co_applicant_age = st.sidebar.multiselect(
        "Select co-applicant age range",
        options=df['co_applicant_age'].unique(),
        default=df['co_applicant_age'].unique(),
    )
    income_min = df['income'].min()
    income_max = df['income'].max()
    income_median = df['income'].median()
    range_width_income = (income_max - income_min)
    default_range_income = (income_median - range_width_income / 4, income_median + range_width_income / 4)
    selected_income = st.sidebar.slider(
        "Select income range",
        income_min,
        income_max,
        default_range_income,
    )
    selected_loan_types = st.sidebar.multiselect(
    "Select loan types", 
    options=df['loan_type'].unique(),
    default=df['loan_type'].unique(),
    )
    loan_amount_min = float(df['loan_amount'].min())
    loan_amount_max = float(df['loan_amount'].max())
    loan_amount_median = float(df['loan_amount'].median())
    range_width_loan_amount = (loan_amount_max - loan_amount_min)
    default_range_loan_amount = (loan_amount_median - range_width_loan_amount / 4, loan_amount_median + range_width_loan_amount / 4)
    selected_loan_amount = st.sidebar.slider(
    "Select loan amount range",
    loan_amount_min,
    loan_amount_max,
    default_range_loan_amount
    )
    
    select_interest_rate = st.sidebar.multiselect(
        "Select interest rate intervals",
        options=df['interest_rate'].unique(),
        default=df['interest_rate'].unique(),
    )
    
    select_total_loan_costs = st.sidebar.multiselect(
        "Select the total loan costs",
        options=df['total_loan_costs'].unique(),
        default=df['total_loan_costs'].unique(),
    )
    
    select_loan_term = st.sidebar.multiselect(
        "Select the loan terms",
        options=df['loan_term'].unique(),
        default=df['loan_term'].unique(),
    )
    property_value_min = df['property_value'].min()
    property_value_max = df['property_value'].max()
    select_property_value = st.sidebar.slider(
        "Select property value",
        property_value_min,
        property_value_max,
        (property_value_min, property_value_max)
    )
    st.write("Occupancy_type")
    select_occupancy_type = st.sidebar.multiselect(
        "Select occupancy type",
        options=df['occupancy_type'].unique(),
        default=df['occupancy_type'].unique(),
    )
    st.write("Total units")
    select_total_units = st.sidebar.multiselect(
        "Select total units",
        options=df['total_units'].unique(),
        default=df['total_units'].unique(),
    )

df_selection = df.query(
    "applicant_ethnicity_1.isin(@selected_applicant_ethnicity) & "
    "applicant_race_1.isin(@selected_applicant_race) & "
    "applicant_sex.isin(@selected_applicant_sex) & "
    "applicant_age.isin(@selected_applicant_age) & "
    "co_applicant_ethnicity_1.isin(@selected_co_applicant_ethnicity) & "
    "co_applicant_race_1.isin(@selected_co_applicant_race) & "
    "co_applicant_sex.isin(@selected_co_applicant_sex) & "
    "co_applicant_age.isin(@selected_co_applicant_age) & "
    "income.between(@selected_income[0], @selected_income[1]) & "
    "loan_type.isin(@selected_loan_types) & "
    "loan_amount.between(@selected_loan_amount[0], @selected_loan_amount[1]) & "
    "interest_rate.isin(@select_interest_rate) & "
    "total_loan_costs.isin(@select_total_loan_costs) & "
    "loan_term.isin(@select_loan_term) & "
    "property_value.between(@select_property_value[0], @select_property_value[1]) & "
    "occupancy_type.isin(@select_occupancy_type) & "
    "total_units.isin(@select_total_units)"
)

# showData=st.multiselect('Filter: ',df_selection.columns, default=['action_taken', 'loan_type', 'lien_status', 'open_end_line_of_credit',
#      'loan_amount', 'combined_loan_to_value_ratio', 'interest_rate', 'total_loan_costs', 'origination_charges', 'loan_term', 
#      'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features', 'property_value', 'occupancy_type',
#       'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest', 'total_units', 'income', 'debt_to_income_ratio', 
#        'applicant_credit_score_type', 'co_applicant_credit_score_type', 'applicant_ethnicity_1', 'co_applicant_ethnicity_1', 'applicant_race_1',
#         'applicant_race_2', 'co_applicant_race_1', 'co_applicant_race_2', 'applicant_sex', 'co_applicant_sex', 'applicant_age', 'co_applicant_age'])



st.dataframe(df_selection, hide_index=True)

column_to_inspect = 'action_taken'
value_counts = df_selection[column_to_inspect].value_counts()

st.write(f"Value Counts for {column_to_inspect}: ")

st.bar_chart(value_counts, color=["#ff6e55"] )
