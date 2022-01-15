import torch
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch import optim
from tqdm import tqdm
import os
import copy

import config
from data_utils import PairDataset
from utils import SampleDataset,collate_fn,ScheduledSampler
from model import PGN

def train(train_iter, model, v, teacher_forcing):
    """
    Args:
        dataset (dataset.PairDataset)
        val_dataset (dataset.PairDataset)
        v (vocab.Vocab)
        start_epoch (int, optional)
    """
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    #
    batch_loss = 0
    batch_losses = []
    num_batches = len(train_iter)
    for batch, data in enumerate(tqdm(train_iter)):
        x, y, x_len, y_len, oov, len_oovs = data
        if config.device=='cuda':
            x = x.to(config.device)
            y = y.to(config.device)
            x_len = x_len.to(config.device)
            y_len = y_len.to(config.device)
            len_oovs = len_oovs.to(config.device)

        loss = model(x,
                     x_len,
                     y,
                     len_oovs,
                     batch=batch,
                     num_batches=num_batches,
                     teacher_forcing=teacher_forcing)

        batch_losses.append(loss.item())
        loss.backward()
        clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(model.attention.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        # batch_loss += loss.item()

    batch_losses = np.mean(batch_losses)
    return batch_losses

def evaluate(model, eval_iter):
    """
    Args:
        model (torch.nn.Module)
        val_data (dataset.PairDataset)
    """
    val_loss = []
    model.eval()
    with torch.no_grad():
        device= config.device
        for batch, data in enumerate(tqdm(eval_iter)):
            x, y, x_len, y_len, oov, len_oovs = data
            if config.device=='cuda':
                x = x.to(device)
                y = y.to(device)
                x_len = x_len.to(device)
                y_len = y_len.to(device)
                len_oovs = len_oovs.to(device)
            loss = model(x,
                         x_len,
                         y,
                         len_oovs,
                         batch=batch,
                         num_batches=len(eval_iter),
                         teacher_forcing=True)
            val_loss.append(loss.item())
    return np.mean(val_loss)

if __name__=='__main__':
    train_dataset = PairDataset(config.train_data_path,
                          max_src_len=config.max_src_len,
                          max_tgt_len=config.max_tgt_len,
                          truncate_src=config.truncate_src,
                          truncate_tgt=config.truncate_tgt)
    dev_dataset = PairDataset(config.dev_data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)
    vocab = train_dataset.build_vocab()
    train_data = SampleDataset(train_dataset.pairs, vocab)
    val_data = SampleDataset(dev_dataset.pairs, vocab)
    train_iter = DataLoader(dataset=train_data,
                            batch_size=config.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
    eval_iter = DataLoader(dataset=val_data,
                           batch_size=config.batch_size,
                           shuffle=True,
                           pin_memory=True, drop_last=True,
                           collate_fn=collate_fn)
    model = PGN(vocab)
    model.load_model()
    model.to(config.device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    best_val_loss = np.inf
    start_epoch = 0
    num_epochs = len(range(start_epoch, config.epoch))
    scheduled_sampler = ScheduledSampler(num_epochs)#生成随机数字[0.0,0.25,...1.0]
    if config.scheduled_sampling:
        print('动态 Teather forcing 模式打开')
    writer = SummaryWriter(config.log_path)

    for epoch in range(start_epoch, config.epoch):
        model.train()
        # Teacher Forcing模式
        if config.scheduled_sampling:
            teacher_forcing = scheduled_sampler.teacher_forcing(epoch - start_epoch)
        else:
            teacher_forcing = True
        print('teacher_forcing = {}'.format(teacher_forcing))
        # 训练
        batch_loss = train(train_iter, model, vocab, teacher_forcing)
        writer.add_scalar('Train/training loss', batch_loss / 10, epoch)
        # 验证
        val_loss = evaluate(model, eval_iter)
        writer.add_scalar('Train/val_loss ', val_loss / 10, epoch)
        print('validation loss:{}'.format(val_loss))
        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(model.encoder, config.encoder_save_name)
            torch.save(model.decoder, config.decoder_save_name)
            torch.save(model.attention, config.attention_save_name)
            torch.save(model.reduce_state, config.reduce_state_save_name)
            # best_model = copy.deepcopy(model.module if hasattr(model, "module") else model)  不知道为啥这样保存predict就会出错
            # torch.save(best_model.state_dict(), os.path.join(config.output_dir, "best_model.pkl"))
            print('save model:',epoch)










