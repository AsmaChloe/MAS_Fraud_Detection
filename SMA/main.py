import pandas as pd
from sklearn.preprocessing import StandardScaler

from SMA.Model import Model

#Importing the dataset
data = pd.read_csv("G:\Mon Drive\S8\TER\Code\creditcard.csv")
data["Amount"] = StandardScaler().fit_transform(data[["Amount"]])
data["Time"] = StandardScaler().fit_transform(data[["Time"]])

#Fraudulent transactions
data_fraud = data[data['Class'] == 1]
#Non fraudulent transactions
data_non_fraud = data[data['Class'] == 0]


nb_round = 100

parameters = [
    # (data_non_fraud, 2, 5, 1, (1,0,0)),
    # (data_non_fraud, 2, 5, 1, (0.5,0.5,0)),
    # (data_non_fraud, 2, 5, 1, (0.5,0,0.5)),
    # (data_non_fraud, 2, 5, 1, (0,0,1)),
    # (data_fraud, 2, 5, 1, (1,0,0)),
    # (data_fraud, 2, 5, 1, (0.5,0.5,0)),
    # (data_fraud, 2, 5, 1, (0.5,0,0.5)),
    # (data_fraud, 2, 5, 1, (0,0,1)),
    (data_fraud, 8, 10, 1, (0,0,1)),
    (data_fraud, 10, 30, 10, (0,0,1)),
    (data_fraud, 20, 60, 10, (0,0,1)),
]

for data, nb_acceptors, nb_proposers, nb_learners, ratio in parameters:
    print(f"Parameters : nb_acceptors = {nb_acceptors}, nb_proposers = {nb_proposers}, nb_learners = {nb_learners}, ratio = {ratio}")

    true_classes = []
    found_classes = []
    steps = []
    crashes = []
    reload = []

    for i in range(nb_round):
        # print(f"Round {i+1}/{nb_round} :")

        sample = data.sample(n=1)
        sample_class = sample['Class'][sample.index[0]]
        sample = sample.drop(['Class'], axis=1)

        model  = Model(nb_proposers, nb_acceptors, nb_learners, data = sample, ratio=ratio)

        while not model.finished :
            model.step()

        nb_steps = model.schedule.steps
        found_class = model.consensus
        nb_crashes = model.nb_crashes
        nb_reload = model.nb_reload

        true_classes.append(sample_class)
        found_classes.append(found_class)
        steps.append(nb_steps)
        crashes.append(nb_crashes)
        reload.append(nb_reload)

        # print(f"True class : {sample_class} | Found cass : {found_class} | Nb steps : {nb_steps} | Nb crashes : {nb_crashes} | Nb reload : {nb_reload}")

    results_df = pd.DataFrame({'true_classes': true_classes,
                               'found_classes': found_classes,
                               'nb_steps': steps,
                               'nb_crashes': crashes,
                               'nb_reload': reload})
    print(results_df.describe())
    print("\n\n")
