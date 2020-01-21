# encoding:utf-8


# embedding
source_dir = "/home1/fangzheng/data/bert_el_data/"

word_embed_path = source_dir + "embedding/word_embeddings.npy"

word_vocab_path = source_dir + "embedding/dict.word"

entity_embed_path = source_dir + "embedding/entity_embeddings.npy"

entity_vocab_path = source_dir + "embedding/dict.entity"

stop_word_path = source_dir + "embedding/stop_words"


# GAT
save_gat_dir = "/home1/fangzheng/data/bert_el_data/gat/old_model/"

gat_train_path = "/home1/fangzheng/data/bert_el_data/aida_train/gat/aida_train_global_graph"

validate_name = "aida_testA"
gat_validate_path = "/home1/fangzheng/data/bert_el_data/" + validate_name + "/gat/" + validate_name + "_global_graph"

test_name = "kore50"
gat_test_path = "/home1/fangzheng/data/bert_el_data/" + test_name + "/gat/" + test_name + "_global_graph"

# the number of candidate entity for each mention
global_candidate_num = 5

# the number of adjacent mentions for each mention in our graph
adjacent_num = 2

# the number of candidate entity in each graph
node_num = global_candidate_num*(adjacent_num+1) + adjacent_num

# the feature size for each node
fea_size = 768

# the number of class for all nodes
class_num = 2

# epoch num for GAT old_model
gat_num_epochs = 100

gat_batch_size = 16

# output loss per batch
print_per_batch = 5

# early stop
require_improvement = 2000

# learning rate
lr = 0.005

# weight decay
l2_coef = 0.0005

# numbers of hidden units per each attention head in each layer
hid_units = [32]

# additional entry for the output layer
n_heads = [8, 1]

residual = False


# Combine old_model
save_combine_dir = "/home1/fangzheng/data/bert_el_data/combine/old_model/"

combine_train_path = "/home1/fangzheng/data/bert_el_data/aida_train/combine/aida_train_combine_vec"

combine_validate_name = "aida_testA"
combine_validate_path = "/home1/fangzheng/data/bert_el_data/" + combine_validate_name + "/combine/" + combine_validate_name + "_combine_vec"

combine_test_name = "ace2004"
combine_test_path = "/home1/fangzheng/data/bert_el_data/" + combine_test_name + "/combine/" + combine_test_name + "_combine_vec"

# epoch num for combine old_model
combine_num_epochs = 10

combine_batch_size = 64

bert_hidden_size = 768

gat_hidden_size = 256

xgboost_fea_size = 12

static_size = 1

combine_learning_rate = 1e-2

combine_mlp_layer = 1

combine_margin = 0.2