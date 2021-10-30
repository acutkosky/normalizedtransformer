

from torchtext.datasets import WikiText2
import transformers
import torch
from torch.utils.data import DataLoader
import argparse
import time
import tqdm
from matplotlib import pyplot as plt
from model import StackedAttention, ModelConfig

parser = argparse.ArgumentParser(description='Simple pretraining thing')

# parser.add_argument('--logdir', default='./runs/', help='Tensorboard logdir')
parser.add_argument('--save_path', default='awesomeplot.png', help='where to save data')
parser.add_argument('--learning_rate', '-lr', type=float, default=6e-4)
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()


class TrainConfig:
    def __init__(self, **kwargs):
        self.batch_size = 64
        self.learning_rate = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epochs = 1


        for k,v in kwargs.items():
            setattr(self, k, v)

def train(model, config):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses = []

    for epoch in range(config.epochs):
        train_iter = WikiText2(root='data/', split='train')
        loader = DataLoader(train_iter, batch_size=config.batch_size)
        pbar = tqdm(enumerate(loader), total=len(loader))
        running_loss = 0.0
        last_time = time.monotonic()

        for t, strings in pbar:
            encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
            idx = encoded['input_ids']
            mask = encoded['attention_mask']
            
            model.zero_grad()
            features, loss = model(idx, mask)
            loss.backward()
            optimizer.step()


            current_time = time.monotonic()
            delta_time = current_time - last_time
            last_time = current_time
            running_loss += (loss.item() - running_loss)/min(it+1.0, 1000.0)
            losses.append(loss.item())
            pbar.set_description(f"epoch {epoch+1} iter {t}: train loss {loss.item():.5f}, running loss {running_loss:0.5f}, accuracy {accuracy:0.5f} speed {1.0/delta_time:0.5f}")


    return losses



def initialize_and_train_model(args):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=10,
        embedding_dim=512,
        n_heads=4,
        layers=args.n_layer)

    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    attention_model = StackedAttention(model_config)

    losses = train(attention_model, train_config)

    plt.plot(losses)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Loss of awesome model!')

    plt.savefig(args.save_path)

    

if __name__=='__main__':
    args = parser.parse_args()
    initialize_and_train_model(args)



    

    


    
