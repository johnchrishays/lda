import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from folktables import ACSDataSource, ACSEmployment

scaler = StandardScaler()

def get_adult_data():
    adult = pd.read_csv('data/adult.data',names = ['age',
        'workclass',
        'fnlwgt',
        'education',
        'educationnum',
        'maritalstatus',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capitalgain',
        'capitalloss',
        'hoursperweek',
        'nativecountry','target'],index_col=False)

    adult['y'] = np.where(adult['target']==' <=50K',0,1)
    adult = adult[['maritalstatus', 'hoursperweek', 'education', 'workclass', 'age', 'fnlwgt', 'race', 'educationnum', 'y']]
    adult = pd.get_dummies(adult, columns=['maritalstatus', 'education', 'workclass', 'race'], drop_first=True)
    adult[['hoursperweek', 'age', 'fnlwgt', 'educationnum']] = scaler.fit_transform(adult[['hoursperweek', 'age', 'fnlwgt', 'educationnum']])

    adult.rename(columns={'race_ White': 'ref_group_indicator'}, inplace=True)

    num_cols = adult.select_dtypes(include="number").columns
    num_cols = [col for col in num_cols if col != "y"]
    adult[num_cols] = scaler.fit_transform(adult[num_cols])
    return adult

def get_folktables_data():
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    acs_features, acs_labels, acs_group = ACSEmployment.df_to_pandas(acs_data)
    acs_group['group'] = acs_group['RAC1P'] == 1
    acs_group.drop(columns=['RAC1P'], inplace=True)
    folktables = pd.DataFrame(acs_features, columns=acs_features.columns)
    folktables['y'] = acs_labels
    folktables['ref_group_indicator'] = acs_group['group']
    num_cols = folktables.select_dtypes(include="number").columns
    folktables[num_cols] = scaler.fit_transform(folktables[num_cols])
    num_cols = [col for col in num_cols if col != "y"]
    return folktables

def get_hmda_data():
    hmda = pd.read_csv('data/2017-NY-features.csv')
    hmda_labels = pd.read_csv('data/2017-NY-target.csv')
    hmda_protected= pd.read_csv('data/2017-NY-protected.csv')

    text_features = hmda.select_dtypes(include=['object']).columns
    hmda['has_co_applicant'] = hmda['has_co_applicant'].fillna(-1)
    hmda = pd.get_dummies(hmda, columns=['has_co_applicant'], drop_first=True)

    if len(text_features) > 0:
        hmda = pd.get_dummies(hmda, columns=text_features, drop_first=True)
    hmda.drop(columns=['sequence_number'], inplace=True)

    hmda['y'] = hmda_labels['action_taken'] == 1
    hmda['ref_group_indicator'] = hmda_protected['applicant_race_1'] == 5
    num_cols = hmda.select_dtypes(include="number").columns
    num_cols = [col for col in num_cols if col != "y"]
    hmda[num_cols] = scaler.fit_transform(hmda[num_cols])
    return hmda

def get_bank_marketing_data():
    bank_marketing = pd.read_csv('data/bankmarketing/bank-additional/bank-additional-full.csv', sep=';')
    bank_marketing['y'] = bank_marketing['y'] == 'yes'
    bank_marketing['ref_group_indicator'] = bank_marketing['marital'] == 'married'
    num_cols = bank_marketing.select_dtypes(include="number").columns
    num_cols = [col for col in num_cols if col != "y"]
    bank_marketing[num_cols] = scaler.fit_transform(bank_marketing[num_cols])
    text_features = bank_marketing.select_dtypes(include=['object']).columns
    text_features = [col for col in text_features if col != "y"]
    bank_marketing = pd.get_dummies(bank_marketing, columns=text_features, drop_first=True)
    return bank_marketing

def get_gmsc_data():
    gmsc = pd.read_csv('data/gmsc/cs-training.csv')
    gmsc['y'] = gmsc['SeriousDlqin2yrs']
    gmsc.drop(columns=['SeriousDlqin2yrs'], inplace=True)
    gmsc['ref_group_indicator'] = gmsc['NumberOfDependents'] == 0
    num_cols = gmsc.select_dtypes(include="number").columns
    num_cols = [col for col in num_cols if col != "y"]
    gmsc[num_cols] = scaler.fit_transform(gmsc[num_cols])
    gmsc.drop(columns=['NumberOfDependents'], inplace=True)
    gmsc.dropna(inplace=True)
    return gmsc