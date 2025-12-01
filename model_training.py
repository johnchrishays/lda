import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os

from get_datasets import get_adult_data, get_folktables_data, get_hmda_data, get_bank_marketing_data, get_gmsc_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


from fairlearn.reductions import ExponentiatedGradient, DemographicParity, ErrorRate
  
B = 5000 # number of models to train

scaler = StandardScaler()

def srg_score(y_pred, ref_indicator):
    threshold = 0.5
    return sum(y_pred[ref_indicator] > threshold) / len(y_pred[ref_indicator]) - sum(y_pred[~ref_indicator] > threshold) / len(y_pred[~ref_indicator])

##### DATASETS #####
adult = get_adult_data()
folktables = get_folktables_data()
hmda = get_hmda_data()
bank_marketing = get_bank_marketing_data()
gmsc = get_gmsc_data()
datasets = [adult, folktables, hmda]
dataset_names = ['adult', 'folktables', 'hmda']

##### METHODS #####
run_with_fairlearn = False
model_classes = ['logistic', 'RF', 'gbt']
if run_with_fairlearn:
    model_classes = ['fl_' + model_class for model_class in model_classes]

##### MODEL TRAINING #####
def train_models(sample_X, sample_y, population_X, population_y, model_classes, B, master_seed=0):
    results = []
    rng = np.random.default_rng(master_seed)
    seeds_data = rng.integers(0, 2**31, size=B)
    seeds_models = rng.integers(0, 2**31, size=B)

    total_run_time = {model_class: 0 for model_class in model_classes}
    
    for b in tqdm(range(B)):
        X_train, X_eval, y_train, y_eval = train_test_split(sample_X, sample_y, test_size=0.3, random_state=seeds_data[b])
        
        it_results = {}
        for model_class in model_classes:
            start_time = time.time()
            if 'logistic' in model_class:
                model = LogisticRegression(max_iter=2400, random_state=seeds_models[b])
            elif 'RF' in model_class:
                model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=seeds_models[b])
            elif 'gbt' in model_class:
                model = HistGradientBoostingClassifier(max_iter=100, random_state=seeds_models[b])
            if 'fl_' in model_class:
                objective = ErrorRate(costs={'fp': 0.5, 'fn': 0.5})
                model = ExponentiatedGradient(model, constraints=DemographicParity(difference_bound=0.2), objective=objective)
                model.fit(X_train, y_train, sensitive_features=X_train["ref_group_indicator"])
            else:
                model.fit(X_train, y_train)
            end_time = time.time()
            total_run_time[model_class] += end_time - start_time
            training_accuracy = accuracy_score(y_train, model.predict(X_train))
            eval_accuracy = accuracy_score(y_eval, model.predict(X_eval))
            df_accuracy = accuracy_score(population_y, model.predict(population_X))

            if b % 10 == 0 and b > 0:
                print(f"Average run time for {model_class}: {total_run_time[model_class] / b:.1f} seconds")

            training_SRG = np.abs(srg_score(model.predict(X_train), X_train["ref_group_indicator"]))
            eval_SRG = np.abs(srg_score(model.predict(X_eval), X_eval["ref_group_indicator"]))
            df_SRG = np.abs(srg_score(model.predict(population_X), population_X["ref_group_indicator"]))

            it_results.update({
                f'{model_class}_training_accuracy': training_accuracy,
                f'{model_class}_eval_accuracy': eval_accuracy,
                f'{model_class}_df_accuracy': df_accuracy,
                f'{model_class}_training_SRG': training_SRG,
                f'{model_class}_eval_SRG': eval_SRG,
                f'{model_class}_df_SRG': df_SRG
            })
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
    for i, dataset in enumerate(datasets):
        print(dataset_names[i])
        sample_dataset = dataset.sample(n=3000, replace=True, random_state=fileno)
        sample_dataset_X = sample_dataset.drop(columns=['y'])
        sample_dataset_y = sample_dataset['y']
        results_df = train_models(sample_dataset_X, sample_dataset_y, dataset.drop(columns=['y']), dataset['y'], model_classes, B, master_seed=fileno)
        if run_with_fairlearn:
            existing_results_path = f"results_data/{dataset_names[i]}_fairlearn_results_{fileno}.csv"
        else:
            existing_results_path = f"results_data/{dataset_names[i]}_results_{fileno}.csv"
        # if os.path.exists(existing_results_path):
        #     existing_results_df = pd.read_csv(existing_results_path)
        #     for col in results_df.columns:
        #         existing_results_df[col] = results_df[col]
        #     results_df = existing_results_df
        results_df.to_csv(existing_results_path, index=False)
