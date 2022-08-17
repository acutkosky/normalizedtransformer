

from torchtext.datasets import WikiText2
import transformers
import torch
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from model import StackedAttention, ModelConfig
import perpopt
import expmd
from datasets import load_dataset, load_from_disk

import wandb

parser = argparse.ArgumentParser(description='Simple pretraining thing')

# parser.add_argument('--logdir', default='./runs/', help='Tensorboard logdir')
parser.add_argument('--save_path', default='awesomeplot.png', help='where to save data')
parser.add_argument('--learning_rate', '-lr', type=float, default=6e-4)
parser.add_argument('--eps', '-eps', type=float, default=6e-4)
parser.add_argument('--n_layers', type=int, default=12)
parser.add_argument('--n_heads', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--ministeps', type=int, default=1)
parser.add_argument('--weight_decay', '-wd', type=float, default=0.0)
parser.add_argument('--use_diag', '-diag', type=str, default='true')
parser.add_argument('--opt', '-opt', type=str, default='adamw', choices=['adamw', 'perpopt', 'expmd', 'expmdnorm'])


args = parser.parse_args()


class TrainConfig:
    def __init__(self, **kwargs):
        self.batch_size = 64
        self.learning_rate = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epochs = 1
        self.wd = 0.0
        opt = 'adamw'


        for k,v in kwargs.items():
            setattr(self, k, v)

def train(model, config, device):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    if config.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.wd)
    elif config.opt == 'perpopt':
        optimizer = perpopt.PerpOpt(model.parameters(), lr=config.learning_rate, wd=config.wd, beta=0.9, alignbeta=1.0)
    elif config.opt == 'expmd':
        optimizer = expmd.ExpMD(model.parameters(), lr=config.learning_rate, wd=config.wd, eps=config.eps)
    elif config.opt == 'expmdnorm':
        optimizer = expmd.ExpMDNorm(model.parameters(), lr=config.learning_rate, wd=config.wd, eps=config.eps)
    losses = []

    for epoch in range(config.epochs):
        ds_path = f"data/manual_saves/wikitext-2-v1/bs-{config.batch_size}-ml-{model.config.context_length}-train"
        try:
            wikitext = load_from_disk(ds_path)
            print("loaded!")
        except FileNotFoundError:   
            print("file not found, generating from scratch")
            wikitext = load_dataset('wikitext', 'wikitext-2-v1', split='train')
            wikitext = wikitext.filter(lambda x: len(x['text']) > 1)
            wikitext.shuffle()
            wikitext = wikitext.map(lambda examples: tokenizer(examples["text"], 
                                                            padding=True,
                                                            truncation=True,
                                                            max_length=model.config.context_length),
                                    remove_columns=["text"],
                                    batched=True,
                                    batch_size=config.batch_size)
            wikitext.save_to_disk(ds_path)
        # wikitext = load_dataset('wikitext', 'wikitext-2-v1', split='train')
        # wikitext = wikitext.filter(lambda x: len(x['text']) > 1)
        # wikitext.shuffle()
        # print(f"cache: {wikitext.cache_files}")
        # wikitext = wikitext.map(lambda examples: tokenizer(examples["text"], 
        #                                                    padding=True,
        #                                                    truncation=True,
        #                                                    max_length=model.config.context_length),
        #                         remove_columns=["text"],
        #                         batched=True,
        #                         batch_size=config.batch_size)
        # print(f"cache: {wikitext.cache_files}")
        loader = DataLoader(wikitext.with_format('torch'), batch_size=config.batch_size)
        # train_iter = WikiText2(root='data/', split='train')
        # loader = DataLoader(train_iter, batch_size=config.batch_size, shuffle=True)
        pbar = tqdm(enumerate(loader))#, total=len(loader))
        running_loss = 0.0
        running_accuracy = 0.0
        last_time = time.monotonic()

        for t, strings in pbar:
            # encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
            idx = strings['input_ids'].to(device)
            mask = strings['attention_mask'].to(device)
            
            model.zero_grad()
            features, loss, accuracy = model(idx, mask)
            loss.backward()
            optimizer.step()


            current_time = time.monotonic()
            delta_time = current_time - last_time
            last_time = current_time
            running_loss += (loss.item() - running_loss)/min(t+1.0, 1000.0)
            running_accuracy += (accuracy.item() - running_accuracy)/min(t+1.0, 1000.0)

            # losses.append(loss.item())
            wandb.log({
                "epoch": epoch,
                "train/loss": loss.item(),
                "train/accuracy": accuracy.item()
            })
            pbar.set_description(f"train epoch {epoch+1} iter {t}: train loss {loss.item():.5f}, running loss {running_loss:0.5f}, running accuracy {running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")

            for _ in range(config.ministeps-1):
                model.zero_grad()
                features, loss, accuracy = model(idx, mask)
                loss.backward()
                optimizer.step()

        print("testing:")
        test(model, config, device, epoch)

    return losses



def test(model, config, device, epoch):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    losses = []

    ds_path = f"data/manual_saves/wikitext-2-v1/bs-{config.batch_size}-ml-{model.config.context_length}-test"
    try:
        wikitext = load_from_disk(ds_path)
        print("loaded!")
    except FileNotFoundError:   
        print("file not found, generating from scratch")
        wikitext = load_dataset('wikitext', 'wikitext-2-v1', split='test')
        wikitext = wikitext.filter(lambda x: len(x['text']) > 1)
        wikitext.shuffle()
        wikitext = wikitext.map(lambda examples: tokenizer(examples["text"], 
                                                        padding=True,
                                                        truncation=True,
                                                        max_length=model.config.context_length),
                                remove_columns=["text"],
                                batched=True,
                                batch_size=config.batch_size)
        wikitext.save_to_disk(ds_path)
    
    loader = DataLoader(wikitext.with_format('torch'), batch_size=config.batch_size)

    # train_iter = WikiText2(root='data/', split='train')
    # loader = DataLoader(train_iter, batch_size=config.batch_size, shuffle=True)
    pbar = tqdm(enumerate(loader))#, total=len(loader))
    running_loss = 0.0
    running_accuracy = 0.0
    last_time = time.monotonic()

    for t, strings in pbar:
        # encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
        idx = strings['input_ids'].to(device)
        mask = strings['attention_mask'].to(device)
        
        features, loss, accuracy = model(idx, mask)


        current_time = time.monotonic()
        delta_time = current_time - last_time
        last_time = current_time
        running_loss += (loss.item() - running_loss)/(t+1.0)
        running_accuracy += (accuracy.item() - running_accuracy)/(t+1.0)

        losses.append(loss.item())
        wandb.log({
            "test/loss": loss.item(),
            "test/accuracy": accuracy.item()
        },
        commit=False)
        pbar.set_description(f"test epoch {epoch+1} iter {t}: train loss {loss.item():.5f}, running loss {running_loss:0.5f}, running accuracy {running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")

    return losses



def initialize_and_train_model(args):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=10,
        embedding_dim=512,
        n_heads=4,
        n_layers=args.n_layers,
        use_diag=args.use_diag)
    print(args.use_diag)

    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate*args.batch_size/args.ministeps,
        epochs=args.epochs,
        wd=args.weight_decay,
        opt=args.opt,
        ministeps=args.ministeps,
        eps=args.eps,
    )

    device = 'cpu'
    if torch.cuda.is_available():            
        device = torch.cuda.current_device()
    
    attention_model = StackedAttention(model_config).to(device)

    losses = train(attention_model, train_config, device)

    # plt.plot(losses)
    # plt.xlabel('iterations')
    # plt.ylabel('loss')
    # plt.title('Loss of awesome model!')

    # plt.savefig(args.save_path)

    

if __name__=='__main__':
    args = parser.parse_args()
    wandb.init()
    wandb.config.update(args)
    initialize_and_train_model(args)



    

    


    
