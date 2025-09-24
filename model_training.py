import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from folktables import ACSDataSource, ACSEmployment

from ucimlrepo import fetch_ucirepo 
  
B = 10 # number of bootstrap samples

scaler = StandardScaler()

def srg_score(y_pred, ref_indicator):
    threshold = 0.5
    return sum(y_pred[ref_indicator] > threshold) / len(y_pred[ref_indicator]) - sum(y_pred[~ref_indicator] > threshold) / len(y_pred[~ref_indicator])

##### DATASETS #####
# Adult
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

# Folktables
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["AL"], download=True)
acs_features, acs_labels, acs_group = ACSEmployment.df_to_pandas(acs_data)
acs_group['group'] = acs_group['RAC1P'] == 1
acs_group.drop(columns=['RAC1P'], inplace=True)
folktables = pd.DataFrame(acs_features, columns=acs_features.columns)
folktables['y'] = acs_labels
folktables['ref_group_indicator'] = acs_group['group']

# HMDA
hmda = pd.read_csv('data/2017-NY-features.csv')
hmda_labels = pd.read_csv('data/2017-NY-target.csv')
hmda_protected= pd.read_csv('data/2017-NY-protected.csv')

hmda = hmda[['owner_occupancy', 'loan_amount_000s', 'msamd', 'applicant_income_000s', 'hud_median_family_income', 'tract_to_msamd_income', 'has_co_applicant', 'population', 'minority_population', 'number_of_owner_occupied_units', 'number_of_1_to_4_family_units']]
hmda['y'] = hmda_labels['action_taken'] == 1
hmda['ref_group_indicator'] = hmda_protected['applicant_race_1'] == 5
hmda = pd.get_dummies(hmda, columns=['owner_occupancy', 'has_co_applicant'], drop_first=True)
hmda[['loan_amount_000s', 'applicant_income_000s', 'hud_median_family_income', 'tract_to_msamd_income', 'population', 'minority_population', 'number_of_owner_occupied_units', 'number_of_1_to_4_family_units']] = scaler.fit_transform(hmda[['loan_amount_000s', 'applicant_income_000s', 'hud_median_family_income', 'tract_to_msamd_income', 'population', 'minority_population', 'number_of_owner_occupied_units', 'number_of_1_to_4_family_units']])

model_classes = ['logistic', 'RF', 'nn']

##### MODEL TRAINING #####

def train_models(sample_X, sample_y, population_X, population_y, model_classes, B):
    results = []
    for seed in tqdm(range(B)):
        X_train, X_eval, y_train, y_eval = train_test_split(sample_X, sample_y, test_size=0.5, random_state=seed)
        
        it_results = {}
        neg_flag = False
        for model_class in model_classes:
            if model_class == 'logistic':
                model = LogisticRegression()
            elif model_class == 'RF':
                model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=seed)
            elif model_class == 'nn':
                model = MLPClassifier(hidden_layer_sizes=(25,), max_iter=600, random_state=seed)
            model.fit(X_train, y_train)
            training_accuracy = accuracy_score(y_train, model.predict(X_train))
            eval_accuracy = accuracy_score(y_eval, model.predict(X_eval))
            df_accuracy = accuracy_score(population_y, model.predict(population_X))

            training_SRG = srg_score(model.predict(X_train), X_train["ref_group_indicator"])
            eval_SRG = srg_score(model.predict(X_eval), X_eval["ref_group_indicator"])
            df_SRG = srg_score(model.predict(population_X), population_X["ref_group_indicator"])
            if eval_SRG <= 0:
                neg_flag = True
            if df_SRG <= 0:
                neg_flag = True

            it_results.update({
                f'{model_class}_training_accuracy': training_accuracy,
                f'{model_class}_eval_accuracy': eval_accuracy,
                f'{model_class}_df_accuracy': df_accuracy,
                f'{model_class}_training_SRG': training_SRG,
                f'{model_class}_eval_SRG': eval_SRG,
                f'{model_class}_df_SRG': df_SRG
            })
        if neg_flag:
            continue
        results.append(it_results)
    results_df = pd.DataFrame(results)
    return results_df
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, required=False, dest='fileno')
    args = parser.parse_args()
    fileno = args.fileno
    if fileno is None:
        fileno = 0
    datasets = [adult, folktables, hmda]
    dataset_names = ['adult', 'folktables', 'hmda']
    for i, dataset in enumerate(datasets):
        print(dataset_names[i])
        sample_dataset = dataset.sample(n=3000, replace=True)
        sample_dataset_X = sample_dataset.drop(columns=['y'])
        sample_dataset_y = sample_dataset['y']
        results_df = train_models(sample_dataset_X, sample_dataset_y, dataset.drop(columns=['y']), dataset['y'], model_classes, B)
        results_df.to_csv(f"results_data/{dataset_names[i]}_results_{fileno}.csv", index=False)