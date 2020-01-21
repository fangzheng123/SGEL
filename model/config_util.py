# encoding:utf-8


# embedding
source_dir = "/data/fangzheng/bert_el/"

word_embed_path = source_dir + "embedding/word_embeddings.npy"

word_vocab_path = source_dir + "embedding/dict.word"

entity_embed_path = source_dir + "embedding/entity_embeddings.npy"

entity_vocab_path = source_dir + "embedding/dict.entity"

stop_word_path = source_dir + "embedding/stop_words"


# Selector
train_name = "wiki_clueweb"

combine_train_path = source_dir + train_name + "/bert/" + train_name + "_cut_rank_format_bert"

save_combine_dir = source_dir + "model/selector/"

validate_name = "aida_testB"
combine_val_path = source_dir + validate_name + "/bert/" + validate_name + "_cut_rank_format_bert"

test_name = "reuters128"
combine_test_path = source_dir + test_name + "/bert/" + test_name + "_cut_rank_format_bert"

combine_batch_size = 1

# the number of candidate entity for each mention
candidate_num = 5

sequence_len = 5

# the number of candidate entity in each graph
node_num = candidate_num*sequence_len

# epoch num for combine model
combine_num_epochs = 20

# numbers of hidden units per each attention head in each layer
hid_units = [32, 8]

# additional entry for the output layer
n_heads = [8, 1]

residual = False

require_improvement = 3000

bert_hidden_size = 768

gat_hidden_size = 256

xgboost_fea_size = 11

static_size = 1

combine_learning_rate = 1e-3

combine_mlp_layer = 2

combine_margin = 0.3