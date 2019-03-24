import torch.nn.functional as F
from model import CharacterRNN
import torch
from data_process import *

CUDA = torch.cuda.is_available()


def predict(net, char, h=None, top_k=None):
        x = np.array([[net.char2int[char]]])
        x = onehot(x, len(net.chars))
        inputs = torch.from_numpy(x)
        if CUDA:
            inputs = inputs.cuda()
        h = tuple([each.data for each in h])
        out, h = net(inputs, h)

        p = F.softmax(out, dim=1).data
        p = p.cpu()

        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
            
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())

        return net.int2char[char], h


def sample(net, size, prime='The', top_k=None):
    if CUDA:
        net.cuda()
    
    net.eval()
    
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


def main(model_path, num_chars, prime, top_k):
    DATA_FILE = 'data/data.txt'
    with open(DATA_FILE, 'r') as f:
        text = f.read()
    char, int2char, char2int, encoded = encode_text(text)

    checkpoint = torch.load(model_path)
    char = checkpoint['tokens']
    n_hidden = checkpoint['n_hidden']
    n_layers = checkpoint['n_layers']

    net = CharacterRNN(char, n_hidden, n_layers)
    net.load_state_dict(checkpoint['state_dict'])
    print(sample(net, num_chars, prime='He ', top_k=5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to trained model", default='./trained_model/char_rnn_model.net')
    parser.add_argument("--num_chars", help="no of characters to generate", default=1000)
    parser.add_argument("--prime", help="Starting chars ", default="the")
    parser.add_argument("--top_k", help="choose top k chars", default=5)
    args = parser.parse_args()
    main(args.model_path, args.num_chars, args.prime, args.top_k)