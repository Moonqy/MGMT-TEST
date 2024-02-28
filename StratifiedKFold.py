import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()



train = pd.read_csv("your.csv")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = []
targets = ["MGMT statue"]
target="MGMT status"

for fold, (trn_idx, val_idx) in enumerate(
    skf.split(train, train[target])
):
    train.loc[val_idx, "fold"] = int(fold)


train.to_csv("your-train.csv", index=False)
