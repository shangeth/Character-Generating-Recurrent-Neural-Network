from model import CharacterRNN
from data_process import *
import argparse
import torch
import torch.nn as nn

CUDA = torch.cuda.is_available()


def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):

    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    if CUDA:
        net.cuda()
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        h = net.init_hidden(batch_size)
        
        for x, y in batcher(data, batch_size, seq_length):
            counter += 1
            x = onehot(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            if CUDA:
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            net.zero_grad()
            
            output, h = net(inputs, h)
            
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in batcher(val_data, batch_size, seq_length):
                    x = onehot(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if CUDA:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                net.train()
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Train Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

def save_model():

    model_name = 'character_rnn.net'
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': net.chars}

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)

def main(data_preprocess, batch_size, n_epochs, seq_length, lr, print_every, save_path):
    if data_preprocess:
        DATA_FILE = 'data/data.txt'
        with open(DATA_FILE, 'r') as f:
            text = f.read()
        char, int2char, char2int, encoded = encode_text(text)

    n_hidden=512
    n_layers=2
    net = CharacterRNN(char, n_hidden, n_layers)
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=lr, print_every=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_preprocess", help="Preprocess the data?", default=True)
    parser.add_argument("--batch_size", help="no of sequence in a batch", default=20)
    parser.add_argument("--epochs", help="no of epochs to train", default=50)
    parser.add_argument("--seq_len", help="no of character in a sequence", default=100)
    parser.add_argument("--lr", help="learning rate", default=0.001)
    parser.add_argument("--print_every", help="show loss every n steps", default=1)
    parser.add_argument("--save_path", help="path to save the model", default='./trained_model')
    args = parser.parse_args()
    main(args.data_preprocess, args.batch_size, args.epochs, args.seq_len, args.lr, args.print_every, args.save_path)