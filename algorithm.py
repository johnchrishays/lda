import argparse
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

datasets = ['adult', 'folktables', 'hmda']
dataset_names = {'adult': 'Adult', 'folktables': 'Folktables', 'hmda': 'HMDA'}

model_classes = ['logistic', 'RF', 'mlp', 'gbt']
# model_classes = model_classes + ['fl_' + model_class for model_class in model_classes]

def compute_expected_marginal_gain(df, u, true_u, model_class):
    filtered_rows = df[df[f'{model_class}_eval_SRG'] < u]
    if len(filtered_rows) == 0:
        return 0
    return (true_u - filtered_rows[f'{model_class}_df_SRG'].mean()) * len(filtered_rows) / len(df)


def compute_no_assm_and_true_cei(df, B, T, model_class, i):

    no_assm_df = pd.DataFrame(index=range(B), columns=[f'bound_no_assm_{t}' for t in range(T)])
    true_cei_df = pd.DataFrame(index=range(B), columns=[f'true_cei_{t}' for t in range(T)])


    miscovered = 0
    for b in tqdm(range(B)):
        u_hat_t = 1
        bound_no_assm = []
        true_cei = []

        resampled_df = df.sample(frac=1, replace=True).reset_index(drop=True)
        miscovered_i = False

        for t in range(T):
            if t == 0:
                p_t = 1 - math.exp(-1/delta)
            else:
                p_t = 1 - (t/delta + 1)**(-1/t)
            i_t = i_t if u_hat_t < resampled_df.loc[t, f"{model_class}_eval_SRG"] else t
            u_hat_t = resampled_df.loc[i_t, f"{model_class}_eval_SRG"]
            u_t = resampled_df.loc[i_t, f"{model_class}_df_SRG"]
            bound_no_assm.append(u_hat_t * p_t)
            true_marginal_gain = compute_expected_marginal_gain(df, u_hat_t, u_t, model_class)
            true_cei.append(true_marginal_gain)
            if u_hat_t * p_t < true_marginal_gain and not miscovered_i:
                miscovered += 1
                miscovered_i = True
        no_assm_df.loc[b] = bound_no_assm
        true_cei_df.loc[b] = true_cei

    no_assm_means = no_assm_df.mean(axis=0)
    true_cei_means = true_cei_df.mean(axis=0)

    no_assm_sds = no_assm_df.std(axis=0)
    true_cei_sds = true_cei_df.std(axis=0)

    saved_data = {
        "no_assm_means": no_assm_means.to_list(),
        "true_cei_means": true_cei_means.to_list(),
        "no_assm_sds": no_assm_sds.to_list(),
        "true_cei_sds": true_cei_sds.to_list(),
        "miscovered": miscovered/B,
    }
    saved_data = pd.DataFrame(saved_data)
    # file_path = f'algo_data/{dataset}_{model_class}_{i}.csv'
    # if os.path.exists(file_path):
    #     existing_data = pd.read_csv(file_path, index_col=0)
    #     for col in saved_data.columns:
    #         existing_data[col] = saved_data[col]
    #     saved_data = existing_data
    saved_data.to_csv(f'algo_data/{dataset}_{model_class}_{i}.csv')
        
def compute_mrl_assm_and_true_cei(df, B, T, model_class, i):

    mrl_assm_df = pd.DataFrame(index=range(B), columns=[f'bound_mrl_assm_{t}' for t in range(T)])
    true_cei_df = pd.DataFrame(index=range(B), columns=[f'true_cei_{t}' for t in range(T)])


    T_1 = math.ceil(18 * math.log(3/delta))
    for b in range(B):
        u_hat_t = 1
        bound_mrl_assm = []
        true_cei = []

        resampled_df = df.sample(frac=1, replace=True).reset_index(drop=True)
        quantile_samples = [resampled_df.loc[t, f"{model_class}_eval_SRG"] for t in range(T_1)]
        C = sorted(quantile_samples)[math.floor(T_1/3)]
        S_t = []
        nuhats = []
        lambdas = [1/2]
        mubar_eb = 1
        Deltas = []
        for t in range(T_1, T + T_1):
            if t == T_1:
                p_t = 1 - math.exp(-1/delta)
            else:
                p_t = 1 - ((t - T_1)/delta+ 1)**(-1/(t - T_1))
            i_t = i_t if u_hat_t < resampled_df.loc[t, f"{model_class}_eval_SRG"] else t
            q_hat_t = resampled_df.loc[t, f"{model_class}_eval_SRG"]
            if C - q_hat_t > 0:
                Deltas.append(C - q_hat_t)
                S_t.append(t - T_1)
                nuhats.append((1/2 + sum([Deltas[i] for i,s in enumerate(S_t)])) / (1 + len(S_t)))
            u_hat_t = resampled_df.loc[i_t, f"{model_class}_eval_SRG"]

            if len(S_t) > 1:
                lambdas.append( min(1/2, math.sqrt((2 * math.log(6/delta)) / (sigmasq_hat_tm1 * len(S_t) * math.log(1 + len(S_t))))) )
                mubar_eb_num = (math.log(6/delta) + sum([lambdas[i]*Deltas[i] - (Deltas[i] - (nuhats[i-1] if i > 0 else 1/2))**2 * (math.log(1-lambdas[i]) + lambdas[i]) for i, s in enumerate(S_t)]))
                mubar_eb_denom = (sum([lambdas[i] for i,s in enumerate(S_t)]))
                mubar_eb = mubar_eb_num / mubar_eb_denom

            if len(S_t) > 0:
                sigmasq_hat_tm1 = (1/4 + sum([(Deltas[i] - nuhats[i])**2 for i, s in enumerate(S_t)])) / (1 + len(S_t))


            mu_bar_t = min(u_hat_t, mubar_eb)
            u_t = resampled_df.loc[i_t, f"{model_class}_df_SRG"]
            bound_mrl_assm.append(mu_bar_t * p_t)
            true_cei.append(compute_expected_marginal_gain(df, u_hat_t, u_t, model_class))
        mrl_assm_df.loc[b] = bound_mrl_assm
        true_cei_df.loc[b] = true_cei

    mrl_assm_means = mrl_assm_df.mean(axis=0)
    true_cei_means = true_cei_df.mean(axis=0)

    mrl_assm_sds = mrl_assm_df.std(axis=0)
    true_cei_sds = true_cei_df.std(axis=0)

    saved_data = {
        "mrl_assm_means": mrl_assm_means.to_list(),
        "true_cei_means": true_cei_means.to_list(),
        "mrl_assm_sds": mrl_assm_sds.to_list(),
        "true_cei_sds": true_cei_sds.to_list(),
    }
    saved_data = pd.DataFrame(saved_data)
    saved_data.to_csv(f'algo_data/{dataset}_{model_class}_mrl_{i}.csv')

if __name__ == "__main__":
    delta = 0.05

    B = 1000
    T = 100 # number of model training iterations
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, required=False, dest='fileno')
    args = parser.parse_args()
    fileno = args.fileno
    if fileno is None:
        fileno = 0
    for dataset in datasets:
        print(dataset)
        for model_class in model_classes:
            print(model_class)
            results = pd.read_csv(f'results_data/{dataset}_results_{fileno}.csv')
            compute_no_assm_and_true_cei(results, B, T, model_class, fileno)
            # compute_mrl_assm_and_true_cei(results, B, T, model_class, fileno)
