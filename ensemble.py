import train_xgb
import train_mlp
from sklearn.model_selection import StratifiedKFold
from data_preprocessing import X, y
import math
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def full_training(nb_iter, cv=5, use_xgb=True, use_mlp=True, save_outputs=False):
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []
    for iter in tqdm(range(nb_iter)):
        params_xgb = train_xgb.random_params(train_xgb.param_grid)
        params_mlp = train_mlp.random_params(train_mlp.param_grid)
        full_outputs_xgb = []
        full_outputs_mlp = []
        full_labels = []
        mean_best_epoch = 0
        for train_idx, val_idx in kf.split(X, y):
            X_train = X.iloc[train_idx]
            X_val   = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val   = y.iloc[val_idx]

            if use_xgb:
                full_outputs_xgb.append(train_xgb.train(params_xgb, X_train, y_train, X_val))
            if use_mlp:
                loss, outputs, epoch = train_mlp.train(params_mlp, X_train, y_train, X_val, y_val)
                full_outputs_mlp.append(outputs)
                mean_best_epoch += epoch
            full_labels.append(y_val)
        full_labels = np.concatenate(full_labels)
        mean_best_epoch /= cv
        if use_xgb:
            full_outputs_xgb = np.concatenate(full_outputs_xgb)
        if use_mlp:
            full_outputs_mlp = np.concatenate(full_outputs_mlp)
        if use_xgb and not use_mlp:
            score = roc_auc_score(full_labels, full_outputs_xgb)
            results.append({"Params": params_xgb, "Outputs": full_outputs_xgb, "Score": score})
        if use_mlp and not use_xgb:
            score = roc_auc_score(full_labels, full_outputs_mlp)
            results.append({"Params": params_mlp, "Outputs": full_outputs_mlp, "Score": score, "Mean best epoch": round(mean_best_epoch)})
        if use_xgb and use_mlp:
            best_alpha = 0
            best_score = -math.inf
            for alpha in np.linspace(0, 1, 101):
                p = alpha * full_outputs_xgb + (1 - alpha) * full_outputs_mlp
                score = roc_auc_score(full_labels, p)
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            results.append({
                "Score": best_score, 
                "Alpha": best_alpha, 
                "Params XGB": params_xgb, 
                "Params MLP": params_mlp, 
                "Mean best epoch": round(mean_best_epoch)})
    if results != []:
        results.sort(key=lambda x: x["Score"], reverse=True)
    if use_xgb and not use_mlp and save_outputs:
        joblib.dump(results, "results1_xgb.joblib")
    if use_mlp and not use_xgb and save_outputs:
        joblib.dump(results, "results1_mlp.joblib")
    if use_xgb and use_mlp and save_outputs:
        joblib.dump(results, "results_both.joblib")
    if not use_xgb and not use_mlp:
        return full_labels
    return results

def ensemble_from_outputs(outputs_xgb, outputs_mlp, full_labels):
    results = []
    for output_xgb in tqdm(outputs_xgb[:10]):
        for output_mlp in outputs_mlp[:10]:
            best_alpha = 0
            best_score = -math.inf
            for alpha in np.linspace(0, 1, 101):
                p = alpha * output_xgb["Outputs"] + (1 - alpha) * output_mlp["Outputs"]
                score = roc_auc_score(full_labels, p)
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            results.append({
                "Score": best_score, 
                "Alpha": best_alpha, 
                "Params XGB": output_xgb["Params"], 
                "Params MLP": output_mlp["Params"], 
                "Mean best epoch": output_mlp["Mean best epoch"]})
    results.sort(key=lambda x: x["Score"], reverse=True)
    joblib.dump(results, "ensemble_results.joblib")
    return results

def final_training():
    results = joblib.load("ensemble_results.joblib")
    best_result = results[0]
    params_xgb = best_result["Params XGB"]
    model1 = XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    pipeline = Pipeline(steps=[('preprocessor', params_xgb["preprocessor"]), ('model', model1)])
    pipeline.set_params(**params_xgb)
    pipeline.fit(X, y)

    params_mlp = best_result["Params MLP"]
    X_train_pre = params_mlp["preprocessing"].fit_transform(X)
    X_train_tensor = torch.tensor(X_train_pre, dtype=torch.float32)
    y_train_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)
    train_set = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_set, batch_size=params_mlp["batch_size"])

    input_s = len(X_train_tensor[0])
    model = train_mlp.MLP(input_s, params_mlp["hidden_sizes"], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params_mlp["learning_rate"], weight_decay=params_mlp["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(best_result["Mean best epoch"]):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).flatten()
            loss = criterion(outputs, y_batch)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

    test_data = pd.read_csv('test.csv')
    test_data["GroupId"] = test_data["PassengerId"].str.split("_", expand=True)[0]
    test_data["GroupSize"] = test_data.groupby("GroupId")["GroupId"].transform("size")
    test_data = test_data.drop(["GroupId"], axis=1)
    test_data = test_data.set_index("PassengerId")
    test_data = test_data.drop(["Name"], axis=1)
    test_data[["Cabin deck", "Cabin number", "Cabin side"]] = test_data["Cabin"].str.split("/", expand=True)
    test_data = test_data.drop(["Cabin"], axis=1)

    test_data["Cabin number"] = test_data["Cabin number"].astype(float)
    test_data["CryoSleep"] = test_data["CryoSleep"].astype(float)
    test_data["VIP"] = test_data["VIP"].astype(float)

    probas_xgb = pipeline.predict_proba(test_data)[:, 1]
    test_final = params_mlp["preprocessing"].transform(test_data)
    X_test_final = torch.tensor(test_final,dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(X_test_final).flatten()
        probas_mlp = torch.sigmoid(logits).cpu().numpy()
    final_probas = best_result["Alpha"]*probas_xgb + (1-best_result["Alpha"])*probas_mlp
    final_preds = [True if x >= 0.5 else False for x in final_probas]
    output = pd.DataFrame({'PassengerId': test_data.index, 'Transported': final_preds})
    output.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    """full_labels = full_training(1, use_xgb=False, use_mlp=False, save_outputs=False)
    outputs_xgb = joblib.load("results_xgb.joblib")
    outputs_mlp = joblib.load("results_mlp.joblib")
    results = ensemble_from_outputs(outputs_xgb, outputs_mlp, full_labels)
    print([x["Score"] for x in results[:10]])"""
    final_training()