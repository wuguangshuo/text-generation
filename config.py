import torch


device='cuda' if torch.cuda.is_available() else 'cpu'
train_data_path='./data/train.txt'
dev_data_path='./data/dev.txt'
test_data_path='./data/test.txt'
stop_word_file = './data/HIT_stop_words.txt'
max_src_len=300#文本截断长度
max_tgt_len=100#摘要截断长度
truncate_src=True#文本是否截断
truncate_tgt=True#摘要是否截断

min_dec_steps: int = 30#predict最少生成数目
max_dec_steps: int = 500#predict最多生成数目

max_vocab_size=3000#词典大小
batch_size=64
epoch=8
hidden_size=512
embed_size=512#

learning_rate = 0.001
max_grad_norm = 2.0

beam_size = 3
alpha = 0.2#length
beta = 0.2#coverage
gamma = 1000#eos

pointer=True
coverage=True
fine_tune=False
scheduled_sampling=True
weight_tying=False
if pointer:
    if coverage:
        if fine_tune:#微调
            model_name = 'ft_pgn'
        else:
            model_name = 'cov_pgn'
    elif scheduled_sampling:#teacher forcing
        model_name = 'ss_pgn'
    elif weight_tying:#解码器编码器共享enbedding矩阵
        model_name = 'wt_pgn'
else:
    model_name = 'baseline'

eps = 1e-31
LAMBDA=1#控制损失函数的超参数

# output_dir = "./save_model"
encoder_save_name = './save_model/encoder.pt'
decoder_save_name = './save_model/decoder.pt'
attention_save_name = './save_model/attention.pt'
reduce_state_save_name = './save_model/reduce_state.pt'
log_path='./runs'