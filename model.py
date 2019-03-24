import torch
import torch.nn as nn

CUDA = torch.cuda.is_available()

class CharacterRNN(nn.Module):
  def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
    super(CharacterRNN, self).__init__()
    self.drop_prob = drop_prob
    self.n_layers = n_layers
    self.n_hidden = n_hidden
    self.lr = lr

    self.chars = tokens
    self.int2char = dict(enumerate(self.chars))
    self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        
    self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)    
    self.dropout = nn.Dropout(drop_prob)
    self.fc1 = nn.Linear(n_hidden, 128)
    self.fc2 = nn.Linear(128, len(self.chars))
    self.relu = nn.ReLU()
    
  def forward(self, x, hidden):
    r_output, hidden = self.lstm(x, hidden)
    out = self.dropout(r_output)
    out = out.contiguous().view(-1, self.n_hidden)
    out = self.fc1(out)
    out = self.relu(out)
    out = self.fc2(out)
    return out, hidden
  
  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    if CUDA:
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
              weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
    else:
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
              weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
    return hidden