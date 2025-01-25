import json, os, re, ast
import pickle, utils
import torch, time
import pandas as pd
from rdkit import Chem
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from generate_data import generate_data_by_label
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    average_precision_score,
)
from tqdm import tqdm

nbits = 1024  # 1024
fpFunc_dict = {}
fpFunc_dict[
    "hashap"
] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)


def evaluate(model_name: str, log_filename: str, y_pred, y_score, y_true):
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    auc = roc_auc_score(y_score=y_score, y_true=y_true)
    aupr = average_precision_score(y_true=y_true, y_score=y_score)
    log_to_file_and_console(
        log_file_name=log_filename,
        fmt="model: {}, acc: {}, recall: {}, precision: {}, f1: {}, auc: {}, aupr: {}".format(
            model_name, acc, recall, precision, f1, auc, aupr
        ),
    )


def metrics(preds, trues):
    y_pred = np.argmax(preds, axis=1)
    roc_auc = roc_auc_score(trues, y_pred)
    acc = accuracy_score(trues, y_pred)
    f1 = f1_score(trues, y_pred, average="binary")
    prec = precision_score(trues, y_pred, average="binary")
    rec = recall_score(trues, y_pred, average="binary")
    aupr = average_precision_score(trues, y_pred)
    return acc, prec, rec, f1, roc_auc, aupr


def log_to_file_and_console(
    log_file_name: str, fmt: str = "", log=None, mode: str = "a+"
):
    """_summary_

    Args:
        fmt (str): _description_
        log (str): _description_
        log_file_name (str): _description_
    """

    print(fmt, log)
    if log_file_name == "":
        return
    log_file = open(log_file_name, mode=mode, encoding="utf-8")
    print(fmt, log, file=log_file)


def get_filename_with_suffix(filename: str, suffix: str):
    filename_format = filename[-4:]
    filename = filename[:-4] + suffix + filename_format
    return filename


def get_dict_from_json_filename(filename: str) -> dict:
    return json.loads(open(filename, "r", encoding="utf-8").read())


def save_data_by_pickle(
    data,
    filename: str,
):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def get_herb_dict():
    dict = utils.get_dict_from_json_filename(
        utils.get_processed_filename("id_dict.json")
    )
    new_dict = {}
    for key, val in dict.items():
        if is_herb(key):
            new_dict[key] = val
    return new_dict


def get_formula_dict():
    dict = utils.get_dict_from_json_filename(
        utils.get_processed_filename("id_dict.json")
    )
    new_dict = {}
    for key, val in dict.items():
        if is_formula(key):
            new_dict[key] = val
    return new_dict


import re


def extract_chinese(text):
    chinese_pattern = re.compile(r"[\u4e00-\u9fa5]")
    return "".join(chinese_pattern.findall(text))


def save_json_filename(dict: dict, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            dict,
            f,
            indent=4,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
        )


def get_data_from_pickle(
    filename: str,
):
    with open(filename, "rb") as f:
        return pickle.load(f)


drug_model = ""


def get_drug_embedding_by_smile(
    smile: str,
    device: torch.device,
    model_name: str = "pretrain_model/simcsesqrt-model",
):
    global drug_model
    drug_model = SentenceTransformer(
        "pretrain_model/simcsesqrt-model", device=torch.device("cuda")
    )
    return drug_model.encode(smile), get_cid_fp_by_smile(smile=smile)


# embedder = SeqVecEmbedder()


def get_protein_embedding_by_seq(
    seq: str,
):
    # embedding = embedder.embed(seq)
    f = drug_model.encode(seq)
    return f.astype(float)


# cell_embedding_dict = get_dict_from_json_filename(
#     './drugcombdb/context_set_m.json')


def get_cell_line_embedding(cell: str, default_embedding):
    if cell not in cell_embedding_dict.keys():
        return default_embedding
    return cell_embedding_dict[cell]


def get_cid_fp_by_smile(smile: str, fp_name: str = "hashap"):
    mol = Chem.MolFromSmiles(smile)
    if mol == None:
        return [1] * nbits
    fp = fpFunc_dict[fp_name](mol)
    return fp


def get_dict_from_df(
    df: pd.DataFrame,
    key_index: int,
    val_index: int,
) -> dict:
    dict = {}
    for index, row in df.iterrows():
        dict[row[key_index]] = row[val_index]
    return dict


def add_time_suffix(
    prefix: str,
):
    return prefix + time.strftime("%Y-%m-%d %H_%M_%S", time.localtime(time.time()))


def get_raw_filename(filename: str):
    return "./raw_data/{}".format(filename)


def get_processed_filename(filename: str):
    return "./processed_data/{}".format(filename)


def contains_chinese(s):
    return bool(re.compile(r"^[\u4e00-\u9fff]+$").match(s))


def contains_digits_and_chinese(s):
    pattern = re.compile(r"[\d]+.*[\u4e00-\u9fa5]+")
    return bool(pattern.search(s))


def is_formula(key: str):
    return key.startswith("formula_")


def is_herb(key: str):
    return contains_chinese(key)


def is_target(key: str):
    return key.startswith("HBTAR")


def is_ingredient(key: str):
    return key.startswith("HBIN")


from tqdm import tqdm


def data_oversample(label_filename: str, generate_num: 2000, edge_thr=0.5):
    (
        generate_node_features,
        generate_adjust_matrix,
        generate_labels,
    ) = generate_data_by_label(label_filename, generate_num)
    label_name = utils.extract_chinese(label_filename)
    id_dict = utils.get_dict_from_json_filename(
        utils.get_processed_filename("id_dict.json")
    )
    formula_id = load_node_id(id_dict)["formula_id"]
    generate_id_formula_dict = {}
    max_formula_id = max(formula_id)
    for i in tqdm(range(len(generate_node_features))):
        max_formula_id += 1
        generate_formula = "formula_" "{}方剂".format(max_formula_id)
        id_dict[generate_formula] = max_formula_id
        generate_id_formula_dict[i] = generate_formula
    herb_id_dict = utils.get_herb_dict()
    reversed_herb_id_dict = {v: k for k, v in herb_id_dict.items()}
    edge_df = pd.read_csv(utils.get_processed_filename("formula_herb_link.csv"))
    new_rows = []
    for i in tqdm(range(generate_adjust_matrix.shape[0])):
        max_formula_id = max(formula_id)
        for j in range(generate_adjust_matrix.shape[1]):
            if generate_adjust_matrix[i][j] > edge_thr:
                new_rows.append(
                    {
                        "item1": generate_id_formula_dict[i],
                        "item2": reversed_herb_id_dict[j],
                    }
                )
    new_edge_df = pd.DataFrame(new_rows)
    edge_df = pd.concat([edge_df, new_edge_df], ignore_index=True)
    label_df = pd.read_csv(label_filename)
    for i, label in enumerate(generate_labels):
        row_df = pd.DataFrame(
            {"formula": [generate_id_formula_dict[i]], "label": [label]}
        )
        label_df = pd.concat([label_df, row_df], ignore_index=True)
    label_df.iloc[:, 0] = label_df.iloc[:, 0].apply(
        lambda x: "formula_" + x if not x.startswith("formula_") else x
    )
    formula_feature_dict = utils.get_dict_from_json_filename(
        utils.get_processed_filename("formula_feature.json")
    )
    for i in tqdm(range(len(generate_node_features))):
        formula_feature_dict[generate_id_formula_dict[i]] = list(
            generate_node_features[i].astype(float)
        )
    utils.save_json_filename(
        formula_feature_dict,
        utils.get_processed_filename("formula_feature.json").replace(
            ".json", "_{}_over.json".format(label_name)
        ),
    )
    utils.save_json_filename(
        id_dict,
        utils.get_processed_filename("id_dict.json").replace(
            ".json", "_{}_over.json".format(label_name)
        ),
    )
    edge_df.to_csv(
        utils.get_processed_filename("formula_herb_link.csv").replace(
            ".csv", "_{}_over.csv".format(label_name)
        ),
        index=False,
    )
    label_df.to_csv(label_filename.replace(".csv", "_over.csv"), index=False)


def get_generate_formula_and_id(generate_node_features, formula_list, id_dict):
    (
        generate_node_features,
        generate_adjust_matrix,
        generate_labels,
    ) = generate_data_by_label()
    # add formula list
    # add id dict
    generate_formula_list = []
    generate_formula_id_list = []
    max_formula_id = max(formula_list)
    for i in range(len(generate_node_features)):
        generate_formula = "generate_{}".format(i)
        generate_formula_list.append(generate_formula)
        max_formula_id += 1
        id_dict[generate_formula] = max_formula_id
        generate_formula_id_list.append(max_formula_id)
    return generate_formula_list, generate_formula_id_list


def load_node_id(id_dict: dict):
    formula_list, formula_id = [], []
    herb_list, herb_id = [], []
    target_list, target_id = [], []
    ingredient_list, ingredient_id = [], []
    for key, val in id_dict.items():
        if is_formula(key):
            formula_list.append(key)
            formula_id.append(val)
        elif is_herb(key):
            herb_list.append(key)
            herb_id.append(val)
        elif is_target(key):
            target_list.append(key)
            target_id.append(val)
        elif is_ingredient(key):
            ingredient_list.append(key)
            ingredient_id.append(val)
        else:
            print("key:{}, type error".format(key))
    return {
        "formula_list": formula_list,
        "formula_id": formula_id,
        "herb_list": herb_list,
        "herb_id": herb_id,
        "target_list": target_list,
        "target_id": target_id,
        "ingredient_list": ingredient_list,
        "ingredient_id": ingredient_id,
    }


def get_ingredient_feature(ingredient_list):
    feature_dict = get_dict_from_json_filename(
        get_processed_filename("ingredient_feature.json")
    )
    feature_list = []
    fp_feature_list = []
    for ingredient in tqdm(ingredient_list):
        feature_list.append(feature_dict[ingredient]["f"])
        fp_feature_list.append(feature_dict[ingredient]["fp"])
    return torch.tensor(feature_list, dtype=torch.float32), torch.tensor(
        fp_feature_list, dtype=torch.float32
    )


def get_herb_feature(herb_list):
    feature_dict = get_dict_from_json_filename(
        get_processed_filename("herb_feature.json")
    )
    feature_list = []
    for herb in tqdm(herb_list):
        feature_list.append(feature_dict[herb]["feature"])
    return torch.tensor(feature_list, dtype=torch.float32)


def get_target_feature(target_list):
    feature_dict = get_dict_from_json_filename(
        get_processed_filename("target_feature.json")
    )
    feature_list = []
    for target in tqdm(target_list):
        feature_list.append(feature_dict[target])
    return torch.tensor(feature_list, dtype=torch.float32)


def get_formula_feature(formula_feature_dict_filename, formula_list):
    feature_dict = get_dict_from_json_filename(
        utils.get_processed_filename(formula_feature_dict_filename)
    )
    feature_list = []
    for formula in tqdm(formula_list):
        formula = utils.remove_starting_digits(formula)
        feature_list.append(feature_dict[formula])
    return torch.tensor(feature_list, dtype=torch.float32)


def get_items_edge(
    filename: str, item1_index: int, item2_index: int, id_dict_filename: str
):
    id_dict = utils.get_dict_from_json_filename(
        get_processed_filename(id_dict_filename)
    )
    df = pd.read_csv(get_processed_filename(filename))
    item1_list = []
    item2_list = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        item1 = row[item1_index]
        item2 = row[item2_index]
        item1_id = id_dict[item1]
        item2_id = id_dict[item2]
        item1_list.append(item1_id)
        item2_list.append(item2_id)
    return torch.tensor([item1_list, item2_list])


def save_df_to_csv(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)


def generate_log_filename(
    mode_name: str,
):
    return (
        "./log/"
        + mode_name
        + time.strftime("%Y-%m-%d %H_%M_%S", time.localtime(time.time()))
        + ".txt"
    )


def get_dict_from_json_str(json_item_str):
    json_item = json.dumps(ast.literal_eval(json_item_str))
    parsed_data = json.loads(json_item)
    return parsed_data

def init_herb_hetro(
    label_filename: str = "label/formula_label_理血.csv",
    id_dict_filename: str = "id_dict.json",
    formula_herb_link_filename: str = "formula_herb_link.csv",
    formula_feature_dict_filename: str = "formula_feature.json",
    save_filename: str = utils.get_processed_filename("comb_data_over.pickle"),
):
    print("try to get hetero data from:{}".format(save_filename))
    if os.path.exists(save_filename):
        print("load exist data from:{}".format(save_filename))
        data = utils.get_data_from_pickle(save_filename)
        return data
    label_df = pd.read_csv(get_processed_filename(label_filename))
    data = HeteroData()
    data["formula"].y = []
    for index, row in label_df.iterrows():
        data["formula"].y.append(row[-1])
    (
        data["formula"].train_mask,
        data["formula"].val_mask,
        data["formula"].test_mask,
    ) = get_data_spilt_index(data["formula"].y)
    data["formula"].y = torch.tensor(data["formula"].y, dtype=int)
    id_dict = utils.get_dict_from_json_filename(
        utils.get_processed_filename(id_dict_filename)
    )
    entity_info_dict = load_node_id(id_dict)
    formula_list = entity_info_dict["formula_list"]
    herb_list = entity_info_dict["herb_list"]
    target_list = entity_info_dict["target_list"]
    ingredient_list = entity_info_dict["ingredient_list"]
    formula_id = entity_info_dict["formula_id"]
    herb_id = entity_info_dict["herb_id"]
    target_id = entity_info_dict["target_id"]
    ingredient_id = entity_info_dict["ingredient_id"]
    data["formula"].node_id = formula_id
    data["herb"].node_id = herb_id
    data["target"].node_id = target_id
    data["ingredient"].node_id = ingredient_id
    data["ingredient"].feature, data["ingredient_fp"].feature = get_ingredient_feature(
        ingredient_list
    )
    data["formula"].formula_list = formula_list
    data["herb"].num_nodes = len(herb_id)
    data["formula"].num_nodes = len(formula_id)
    data["target"].num_nodes = len(target_id)
    data["ingredient"].num_nodes = len(ingredient_id)
    data["ingredient_fp"].num_nodes = len(ingredient_id)

    data["herb"].feature = get_herb_feature(herb_list)
    data["target"].feature = get_target_feature(target_list)
    data["formula"].feature = get_formula_feature(
        formula_feature_dict_filename, formula_list
    )
    data["ingredient", "to", "ingredient_fp"].edge_index = torch.tensor(
        [list(ingredient_id), list(ingredient_id)]
    )
    data["herb", "to", "ingredient"].edge_index = get_items_edge(
        filename="herb_ingredient_link.csv",
        item1_index=0,
        item2_index=1,
        id_dict_filename=id_dict_filename,
    )
    data["formula", "to", "herb"].edge_index = get_items_edge(
        filename=formula_herb_link_filename,
        item1_index=0,
        item2_index=1,
        id_dict_filename=id_dict_filename,
    )
    data["ingredient", "to", "target"].edge_index = get_items_edge(
        filename="ingredient_target_link.csv",
        item1_index=0,
        item2_index=1,
        id_dict_filename=id_dict_filename,
    )

    data = T.ToUndirected()(data)
    utils.save_data_by_pickle(data, save_filename)
    return data


def add_generate_data(
    data: HeteroData, labelfilename: str, formula_list: list, id_dict: dict
):
    (
        generate_node_features,
        generate_adjust_matrix,
        generate_labels,
    ) = generate_data_by_label()
    # add formula list
    # add id dict
    generate_formula_list = []
    max_formula_id = max(formula_list)
    for i in range(len(generate_node_features)):
        generate_formula = "generate_{}".format(i)
        generate_formula_list.append(generate_formula)
        max_formula_id += 1
        id_dict[generate_formula] = max_formula_id
    # add features

    # add y
    # add edge


from sklearn.model_selection import KFold, ShuffleSplit, train_test_split
import random
import numpy as np


def get_data_spilt_index(
    label_list: list,
):
    shuffled_idx = np.array(range(len(label_list)))
    random.shuffle(shuffled_idx)  # 已经被随机打乱
    train_idx = shuffled_idx[: int(0.7 * len(label_list))].tolist()
    val_idx = shuffled_idx[
        int(0.7 * len(label_list)) : int(0.9 * len(label_list))
    ].tolist()
    test_idx = shuffled_idx[int(0.9 * len(label_list)) :].tolist()
    train_mask = sample_mask(train_idx, len(label_list))
    val_mask = sample_mask(val_idx, len(label_list))
    test_mask = sample_mask(test_idx, len(label_list))
    return train_mask, val_mask, test_mask


def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)


def get_list_from_df(df: pd.DataFrame, list_index: int):
    l = []
    for index, row in df.iterrows():
        l.append(row[list_index])
    return l


def remove_starting_digits(s):
    return re.sub(r"^\d*", "", s)


def process_dup_formula():
    folder_path = utils.get_processed_filename("/label")
    for filename in os.listdir(folder_path):
        formula_dict = {}
        is_duplicate = []
        if (
            filename.endswith(".csv")
            and not "over" in filename
            and not "delete" in filename
            and "清热" in filename
            and filename == "formula_label_清热.csv"
        ):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                formula = remove_starting_digits(row[0])
                if formula not in formula_dict.keys():
                    formula_dict[formula] = True
                    is_duplicate.append(False)
                else:
                    is_duplicate.append(True)
            df["is_duplicate"] = is_duplicate
            df = df[df["is_duplicate"] == False].drop(columns="is_duplicate")
            new_filename = file_path.replace(".csv", "_delete.csv")
            df.iloc[:, 0] = df.iloc[:, 0].apply(remove_digits)
            df.to_csv(new_filename, index=False)


def process_dup_formula_herb_link():
    folder_path = utils.get_processed_filename("")
    for filename in os.listdir(folder_path):
        link_dict = {}
        is_duplicate = []
        if (
            filename.endswith(".csv")
            and not "over" in filename
            # and not "delete" in filename
            and "formula_herb_link_no_delete.csv" in filename
        ):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                formula = remove_starting_digits(row[0])
                herb = row[1]
                if formula + herb not in link_dict.keys():
                    link_dict[formula + herb] = True
                    is_duplicate.append(False)
                else:
                    is_duplicate.append(True)
            df["is_duplicate"] = is_duplicate
            df = df[df["is_duplicate"] == False].drop(columns="is_duplicate")
            new_filename = file_path.replace(".csv", "_delete.csv")
            df.iloc[:, 0] = df.iloc[:, 0].apply(remove_digits)
            df.to_csv(new_filename, index=False)
            return


def remove_digits(s):
    return "formula_" + remove_starting_digits(s)


def process_dup_formula_feature():
    folder_path = utils.get_processed_filename("")
    for filename in os.listdir(folder_path):
        formula_dict = {}
        new_dict = {}
        if (
            filename.endswith(".json")
            and not "over" in filename
            and filename.startswith("id_dict.json")
        ):
            file_path = os.path.join(folder_path, filename)
            f_dict = utils.get_dict_from_json_filename(file_path)
            for key, val in f_dict.items():
                formula = remove_starting_digits(key)
                if formula not in formula_dict.keys():
                    formula_dict[formula] = True
                    new_dict["formula_" + formula] = val
            new_filename = file_path.replace(".json", ".json")
            utils.save_json_filename(new_dict, new_filename)
            return


def process_dup_id_dict():
    filename = utils.get_processed_filename("id_dict.json")
    id_dict = get_dict_from_json_filename(filename)
    new_dict = {}
    k = 0
    for key, val in id_dict.items():
        if is_formula(key):
            if key not in new_dict.keys():
                new_dict[key] = k
                k += 1
        else:
            new_dict[key] = val
    new_filename = filename.replace(".json", "_new.json")
    utils.save_json_filename(new_dict, new_filename)


def process_atten():
    test_atten_df = pd.read_csv("./atten/num_of_atten_test.csv")
    edge_df = pd.read_csv("./atten/edge.csv")
    item1_list = []
    item2_list = []
    atten_list = []
    for index, row in tqdm(test_atten_df.iterrows(), total=len(test_atten_df)):
        atten = row[0]
        if index == len(edge_df):
            break
        if float(atten) < 0.7 or atten == 1.0:
            continue
        item1 = edge_df.loc[index][0]
        item2 = edge_df.loc[index][1]
        item1_list.append(item1)
        item2_list.append(item2)
        atten_list.append(atten)
    new_df = pd.DataFrame(
        {"item1": item1_list, "item2": item2_list, "atten": atten_list}
    )
    new_df.to_csv("./atten/big_atten_edge.csv", index=False)


def find_herb_atten(herb_list):
    edge_df = pd.read_csv("atten/edge.csv")
    edge_dict = {}
    edge_index_dict = {}
    if os.path.exists('atten/edge_dict.pickle'):
        edge_dict = utils.get_data_from_pickle('atten/edge_dict.pickle')
        edge_index_dict = utils.get_data_from_pickle('atten/edge_index_dict.pickle')
    else:
        for index, row in tqdm(edge_df.iterrows(), total=len(edge_df)):
            if not is_ingredient(row[1]):
                continue
            if not is_herb(row[0]):
                continue
            if row[0] not in edge_dict.keys():
                edge_dict[row[0]] = []
            edge_dict[row[0]].append(row[1])
            index_key = "{}_{}".format(row[0], row[1])
            if index_key in edge_index_dict.keys():
                continue
            edge_index_dict[index_key] = index
        utils.save_data_by_pickle(edge_dict,'atten/edge_dict.pickle')
        utils.save_data_by_pickle(edge_index_dict,'atten/edge_index_dict.pickle')
    ingredient_df = pd.read_csv("raw_data/Ingredient_info.csv")
    ingredient_dict = utils.get_dict_from_df(ingredient_df, 0, 1)
    first_atten_df = pd.read_csv("atten/num_of_atten_first.csv")
    test_atten_df = pd.read_csv("atten/num_of_atten_test.csv")
    link_herb_list = []
    first_atten_list = []
    test_atten_list = []
    ingredient_list = []
    ingredient_name_list = []
    for herb in herb_list:
        for ingredient in edge_dict[herb]:
            link_herb_list.append(herb)
            ingredient_list.append(ingredient)
            ingredient_name_list.append(ingredient_dict[ingredient])
            index_key = "{}_{}".format(herb, ingredient)
            index = edge_index_dict[index_key]
            first_atten = first_atten_df.iloc[index][0]
            test_atten = test_atten_df.iloc[index][0]
            first_atten_list.append(first_atten)
            test_atten_list.append(test_atten)
    df = pd.DataFrame(
        {
            "herb": link_herb_list,
            "ingredient": ingredient_list,
            "ingredient_name": ingredient_name_list,
            "first": first_atten_list,
            "test": test_atten_list,
        }
    )
    df.to_csv("atten/{}_atten.csv".format(herb_list), index=False)


if __name__ == "__main__":
    # init_herb_hetro_try()
    # process_dup_formula()
    # process_dup_formula_feature()
    # process_dup_formula_herb_link()
    # process_dup_id_dict()
    data_oversample(
        label_filename=utils.get_processed_filename("label/formula_label_清热.csv"),
        generate_num=4000,
        edge_thr=0.8
    )
    # process_atten()
    # find_herb_atten(herb_list=['天麻', '钩藤', '黄芩', '牛膝', '藁本', '菊花', '僵蚕', '蛇莓', '蒺藜', '珍珠母', '水牛角', '白花蛇舌草', '石决明'])
