import utils
import numpy as np
import torch
from dgllife.utils import EarlyStopping
from prettytable import PrettyTable
from sklearn.model_selection import KFold, train_test_split
import random
import os
import argparse
from model import RGAT
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(preds, trues):
    preds = preds.cpu()
    trues = trues.cpu()
    y_pred = np.argmax(preds, axis=1)
    roc_auc = roc_auc_score(trues, y_pred)
    acc = accuracy_score(trues, y_pred)
    f1 = f1_score(trues, y_pred, average="binary")
    prec = precision_score(trues, y_pred, average="binary")
    rec = recall_score(trues, y_pred, average="binary")
    aupr = average_precision_score(trues, y_pred)
    return acc, prec, rec, f1, roc_auc, aupr


def index_to_mask(index, len):
    mask = np.zeros(len, dtype=bool)
    for i in index:
        mask[i] = True
    return mask


def train(
    label_filename: str,
    id_dict_filename: str,
    formula_feature_dict_filename: str,
    formula_herb_link_filename: str,
    in_feats: int,
    hidden_feats: int,
    max_epochs: int,
    split: int,
    lr: int,
):
    graph = utils.init_herb_hetro(
        label_filename=label_filename,
        id_dict_filename=id_dict_filename,
        formula_herb_link_filename=formula_herb_link_filename,
        formula_feature_dict_filename=formula_feature_dict_filename,
        save_filename=utils.get_processed_filename(label_filename).replace(
            ".csv", ".pickle"
        ),
    )
    label_name = utils.extract_chinese(label_filename)
    graph = graph.to(device)
    train_mask, val_mask, test_mask = (
        graph["formula"].train_mask,
        graph["formula"].val_mask,
        graph["formula"].test_mask,
    )
    y = graph["formula"].y
    node_types, edge_types = graph.metadata()
    num_nodes = graph["formula"].feature.shape[0]
    num_relations = len(edge_types)
    init_sizes = [graph[x].feature.shape[1] for x in node_types]
    num_classes = torch.max(graph["formula"].y).item() + 1
    filename = "hgan_herb"
    all_acc = []
    all_prec = []
    all_rec = []
    all_f1 = []
    all_roc_auc = []
    all_aupr = []
    log_filename = utils.generate_log_filename("Herb_{}".format(label_name))
    utils.log_to_file_and_console(
        log_file_name=log_filename,
        fmt="",
        log="true num:{}, total num:{}".format(sum(y), len(y)),
    )
    kf = KFold(n_splits=split, shuffle=True, random_state=2023)
    label_num = len(graph["formula"].y)
    for split, (train_index, test_index) in enumerate(kf.split(graph["formula"].y)):
        test_index, val_index = train_test_split(
            test_index, test_size=0.3, random_state=2023
        )
        train_mask = index_to_mask(train_index, label_num)
        val_mask = index_to_mask(val_index, label_num)
        test_mask = index_to_mask(test_index, label_num)
        stopper = EarlyStopping(mode="higher", filename=filename, patience=100)
        model = RGAT(
            in_feats,
            hidden_feats,
            num_classes,
            num_nodes,
            num_relations,
            init_sizes,
            node_types,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        for epoch in range(max_epochs):
            loss_function = torch.nn.CrossEntropyLoss().to(device)
            model.train()
            forward_type = ""
            if epoch == 0:
                forward_type = "first"
            else:
                forward_type = "train"
            f = model(forward_type, graph)
            loss = loss_function(f[train_mask], y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # validation
            forward_type = "valid"
            acc, prec, rec, f1, roc_auc, aupr = test(
                forward_type, graph, model, val_mask
            )
            e_tables = PrettyTable(["epoch", "acc", "pre", "rec", "f1", "auc", "aupr"])
            e_tables.float_format = ".3"
            row = [epoch, acc, prec, rec, f1, roc_auc, aupr]
            e_tables.add_row(row)
            utils.log_to_file_and_console(log_file_name="", fmt="", log=e_tables)
            early_stop = stopper.step(roc_auc, model)
            if early_stop:
                break
        stopper.load_checkpoint(model)
        forward_type = "test"
        acc, prec, rec, f1, roc_auc, aupr = test(forward_type, graph, model, test_mask)
        e_tables = PrettyTable(["test", "acc", "pre", "rec", "f1", "auc", "aupr"])
        e_tables.float_format = ".3"
        row = ["test", acc, prec, rec, f1, roc_auc, aupr]
        e_tables.add_row(row)
        utils.log_to_file_and_console(log_file_name=log_filename, fmt="", log=e_tables)
        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        all_roc_auc.append(roc_auc)
        all_aupr.append(aupr)
    log_spilt_metrics(log_filename, "acc", all_acc)
    log_spilt_metrics(log_filename, "f1", all_f1)
    log_spilt_metrics(log_filename, "roc_auc", all_roc_auc)
    log_spilt_metrics(log_filename, "aupr", all_aupr)
    log_spilt_metrics(log_filename, "rec", all_rec)
    log_spilt_metrics(log_filename, "prec", all_prec)


def log_spilt_metrics(log_filename, name, all_num):
    all_num = np.array(all_num)
    bit_num = 3
    utils.log_to_file_and_console(
        log_file_name=log_filename,
        fmt="{}: {}({})".format(
            name, round(np.mean(all_num), bit_num), round(np.std(all_num), bit_num)
        ),
        log=None,
    )


def seed_torch(seed: int = 42):
    """_summary_

    Args:
        seed (int, optional): _description_. Defaults to 42.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


@torch.no_grad()
def test(forward_type, graph, model, mask):
    model.eval()
    out = model(forward_type, graph)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[mask], graph["formula"].y[mask])
    _, pred = out.max(dim=1)
    correct = int(pred[mask].eq(graph["formula"].y[mask]).sum().item())
    acc = correct / int(mask.sum())
    return metrics(out[mask], graph["formula"].y[mask])


def run_expriments(
    formula_efficacy_name: str,
    is_over_sample: bool,
    in_feats: int,
    hidden_feats: int,
    max_epochs: int,
    split: int,
    lr: int,
):
    seed_torch()
    is_over_sample = False
    lable_filename = "label/formula_label_{}_over.csv".format(formula_efficacy_name)
    id_dict_filename = "id_dict_{}_over.json".format(formula_efficacy_name)
    formula_feature_filename = "formula_feature_{}_over.json".format(formula_efficacy_name)
    formula_herb_filename = "formula_herb_link_{}_over.csv".format(formula_efficacy_name)
    if not is_over_sample:
        lable_filename = lable_filename.replace("_over", "")
        id_dict_filename = id_dict_filename.replace("_{}_over".format(formula_efficacy_name), "")
        formula_feature_filename = formula_feature_filename.replace(
            "_{}_over".format(formula_efficacy_name), ""
        )
        formula_herb_filename = formula_herb_filename.replace(
            "_{}_over".format(formula_efficacy_name), ""
        )
    train(
        lable_filename,
        id_dict_filename,
        formula_feature_filename,
        formula_herb_filename,
        in_feats,
        hidden_feats,
        max_epochs,
        split,
        lr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCMRGAT training")
    parser.add_argument(
        "--formula_efficacy_name",
        type=str,
        default="理血",
        help="Predicted efficacy name",
    )
    parser.add_argument(
        "--is_over_sample",
        type=bool,
        default=False,
        help="Whether to predict the enhanced data",
    )
    parser.add_argument(
        "--in_feats",
        type=int,
        default=512,
        help="Input feature length of the model",
    )
    parser.add_argument(
        "--hidden_feats",
        type=int,
        default=256,
        help="The hidden layer feature length of the model",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Maximum number of iterations of the model",
    )
    parser.add_argument("--split", type=int, default=10, help="Split of training data")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate in training process"
    )
    args = parser.parse_args()
    run_expriments(
        formula_efficacy_name=args.formula_efficacy_name,
        is_over_sample=args.is_over_sample,
        in_feats=args.in_feats,
        hidden_feats=args.hidden_feats,
        max_epochs=args.max_epochs,
        split=args.split,
        lr=args.lr,
    )
