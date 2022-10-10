import os

import torch
import torch.nn as nn
import pandas as pd
from lstm import CNNLSTM
from utils import MTSDataset

def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")


def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return

    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list
                  }
    torch.save(state_dict, save_path)
    print(f'Metrics saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path)
    print(f'Metrics loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']
def train(model,
          optimizer,
          train_loader,
          valid_loader,
          file_path,
          criterion = nn.CrossEntropyLoss(),
          num_epochs = 5,
          eval_every = None,
          best_valid_loss = float("Inf")):

    if eval_every is None:
        eval_every = len(train_loader) // 2
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    model.train()
    for epoch in range(num_epochs):
        for (values, labels, index) in valid_loader:
            output = model(values.reshape(1, 1, 450).float())

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for (values, labels, index) in valid_loader:
                        output = model(values.reshape(1, 1, 450).float())
                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # reseting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_epochs*len(train_loader)}], Train Loss: {average_train_loss:.4f}, Valid Loss: {average_valid_loss:.4f}")

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)


    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print("Finished Training!")

model = CNNLSTM(450, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
STUDY='4c_lb2020'
FOLD='0'
df_train = pd.read_parquet(os.path.join('data', STUDY, FOLD, 'train-train.parquet')).reset_index(drop=True)
df_val = pd.read_parquet(os.path.join('data', STUDY, FOLD, 'train-val.parquet')).reset_index(drop=True)
df_test = pd.read_parquet(os.path.join('data', STUDY, FOLD, 'test.parquet')).reset_index(drop=True)

train_iter = MTSDataset(df_train)
val_iter = MTSDataset(df_val)
test_iter = MTSDataset(df_test)

fp = f'scratch/classification/{STUDY}/{FOLD}/'
if not os.path.exists(fp):
    os.makedirs(fp)
train(model=model, optimizer=optimizer, num_epochs=10, train_loader=train_iter, valid_loader=val_iter, file_path=fp)
