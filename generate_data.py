import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import utils
from gan_model import GraphDiscriminator, GraphGenerator
import matplotlib.pyplot as plt
import argparse

# 初始化参数
input_dim = 100
hidden_dim = 64

# 训练过程

def train_gan(
    generator,
    discriminator,
    num_epochs,
    real_node_features,
    real_adjust_matrix,
    real_labels,
):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.00063)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)
    d_losses = []
    g_losses = []
    for epoch in range(num_epochs):
        # 训练判别器
        d_optimizer.zero_grad()
        real_node_features_tensor = torch.FloatTensor(real_node_features)
        real_adjust_matrix_tensor = torch.FloatTensor(real_adjust_matrix)
        real_label_tensor = torch.FloatTensor(real_labels)
        real_output = discriminator(
            real_node_features_tensor, real_adjust_matrix_tensor, real_label_tensor
        )
        z = torch.randn(real_node_features_tensor.size(0), input_dim)
        fake_data, fake_edges, fake_labels = generator(z)

        fake_output = discriminator(fake_data, fake_edges, fake_labels)

        d_loss = -torch.mean(torch.log(real_output) + torch.log(1 - fake_output))
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        g_optimizer.zero_grad()
        z = torch.randn(real_node_features_tensor.size(0), input_dim)
        fake_data, fake_edges, fake_labels = generator(z)
        fake_output = discriminator(fake_data, fake_edges, fake_labels)

        g_loss = -torch.mean(torch.log(fake_output))
        g_loss.backward()
        g_optimizer.step()
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        # 打印损失
        if epoch % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}"
            )
    plt.plot(range(num_epochs), d_losses, label='Discriminator Loss')
    plt.plot(range(num_epochs), g_losses, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('gan_loss.png',dpi=300)
    plt.show(block=True)

# num_edges herb的个数
def generate_data(
    data_num,
    real_node_features,
    real_adjust_matrix,
    real_labels,
    num_edges,
    num_labels,
    node_feature_length=23,
    num_formula=2736,
):
    # 创建生成器和判别器
    generator = GraphGenerator(
        input_dim, hidden_dim, node_feature_length, num_edges, num_labels
    )
    discriminator = GraphDiscriminator(
        node_feature_length, hidden_dim, num_edges, num_labels
    )
    train_gan(
        generator,
        discriminator,
        num_epochs=5000,
        real_node_features=real_node_features,
        real_adjust_matrix=real_adjust_matrix,
        real_labels=real_labels,
    )
    z = torch.randn(data_num, input_dim)
    generate_node_features, generate_adjust_matrix, generate_labels = generator(z)
    generate_labels = torch.argmax(generate_labels, dim=1)
    return generate_node_features.detach().numpy(), generate_adjust_matrix, generate_labels.cpu().numpy()


def get_adjust_matrix(herb_dict):
    df = pd.read_csv(utils.get_processed_filename("formula_herb_link.csv"))
    formula_dict = utils.get_formula_dict()
    formula_num = len(formula_dict)
    herb_num = len(herb_dict)
    adjust_matrix = np.zeros([formula_num, herb_num])
    for index, row in df.iterrows():
        formula_id = formula_dict[row[0]]
        herb_id = herb_dict[row[1]]
        adjust_matrix[formula_id][herb_id] = 1
    return adjust_matrix


def get_label_list(label_filename):
    label_list = utils.get_list_from_df(pd.read_csv(label_filename), -1)
    label_array = np.zeros([len(label_list), 2])
    for index, label in enumerate(label_list):
        if label == 1:
            label_array[index][1] = 1
        else:
            label_array[index][0] = 1
    return label_array


import os


def generate_data_by_label(label_filename: str, data_num: int):
    if os.path.exists("generate_node_features_{}".format(data_num)):
        return (
            utils.get_data_from_pickle("generate_node_features_{}".format(data_num)),
            utils.get_data_from_pickle("generate_adjust_matrix_{}".format(data_num)),
            utils.get_data_from_pickle("generate_labels_{}".format(data_num)),
        )
    label_array = get_label_list(label_filename=label_filename)
    herb_dict = utils.get_herb_dict()
    adjust_matrix = get_adjust_matrix(herb_dict)
    formula_features_dict = utils.get_dict_from_json_filename(
        utils.get_processed_filename("formula_feature.json")
    )
    formula_features = []
    for key, value in formula_features_dict.items():
        formula_features.append(value)
    formula_features = np.array(formula_features)
    generate_node_features, generate_adjust_matrix, generate_labels = generate_data(
        data_num=data_num,
        real_node_features=formula_features,
        real_adjust_matrix=adjust_matrix,
        real_labels=label_array,
        num_edges=len(herb_dict),
        num_labels=2,
    )
    print(
        "generate_node_features:{}, shape:{}".format(
            generate_node_features, generate_node_features.shape
        )
    )
    print(
        "generate_adjust_matrix:{}, shape:{}".format(
            generate_adjust_matrix, generate_adjust_matrix.shape
        )
    )
    print(
        "generate_labels:{}, shape:{}, sum:{}".format(
            generate_labels, generate_labels.shape, sum(generate_labels)
        )
    )
    utils.save_data_by_pickle(generate_node_features, "generate_node_features")
    utils.save_data_by_pickle(generate_adjust_matrix, "generate_adjust_matrix")
    utils.save_data_by_pickle(generate_labels, "generate_labels")
    return generate_node_features, generate_adjust_matrix, generate_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCMRGAT Data Enhancement")
    parser.add_argument(
        "--formula_efficacy_name",
        type=str,
        default="理血",
        help="Formula efficacy name need to data enhancement",
    )
    parser.add_argument(
        "--generate_num",
        type=int,
        default=2000,
        help="Number of generated samples",
    )
    parser.add_argument(
        "--edge_thr",
        type=float,
        default=0.8,
        help="The adjacency matrix is used as an edge threshold",
    )
    args = parser.parse_args()
    utils.data_oversample(
        label_filename=utils.get_processed_filename("label/formula_label_{}.csv".format(args.formula_efficacy_name)),
        generate_num=args.generate_num,
        edge_thr=args.edge_thr
    )
