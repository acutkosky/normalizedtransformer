

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
import nigt
from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf
import ds_loader

import wandb

parser = argparse.ArgumentParser(description='Simple pretraining thing')

# parser.add_argument('--logdir', default='./runs/', help='Tensorboard logdir')
parser.add_argument('--save_path', default='awesomeplot.png', help='where to save data')
parser.add_argument('--learning_rate', '-lr', type=float, default=6e-4)
parser.add_argument('--beta', type=float, default=0.99)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--eps', '-eps', type=float, default=6e-4)
parser.add_argument('--n_layers', type=int, default=12)
parser.add_argument('--n_heads', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--ministeps', type=int, default=1)
parser.add_argument('--weight_decay', '-wd', type=float, default=0.0)
parser.add_argument('--use_diag', '-diag', type=str, default='true')
parser.add_argument('--scale_type', type=str, default='random')
parser.add_argument('--recenter', type=str, default='true')
parser.add_argument('--implicit', type=str, default='true')
parser.add_argument('--ol', type=str, default='sgd', choices=['ogd', 'dynamicpf', 'optogd', 'optadam'])
parser.add_argument('--adaptive', type=str, default='true')
parser.add_argument('--config', type=str, default='configs/conf.yaml')
parser.add_argument('--opt', '-opt', type=str, default='adamw', choices=['adamw', 'perpopt', 'expmd',
                                    'expmdnorm', 'nigt', 'dynamic', 'nigt_lamb', 'dynamic_reg', 'dynamic_reg_reset',
                                    'random','sgd', 'randomol'])


args = parser.parse_args()
args = OmegaConf.create(vars(args))
if args.config is not None:
    conf = OmegaConf.load(args.config)
    args = OmegaConf.merge(args, conf)

torch.manual_seed(args.manual_seed)

args_dict = OmegaConf.to_container(args)

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
        optimizer = perpopt.PerpOpt(model.parameters(), lr=config.learning_rate, wd=config.wd, beta=config.beta, alignbeta=1.0)
    elif config.opt == 'expmd':
        optimizer = expmd.ExpMD(model.parameters(), lr=config.learning_rate, wd=config.wd, eps=config.eps)
    elif config.opt == 'expmdnorm':
        optimizer = expmd.ExpMDNorm(model.parameters(), lr=config.learning_rate, wd=config.wd, eps=config.eps)
    elif config.opt == 'nigt':
        optimizer = nigt.Nigt(model.parameters(), lr=config.learning_rate, wd=config.wd, beta=config.beta)
    elif config.opt == 'nigt_lamb':
        optimizer = nigt.Nigt(model.parameters(), lr=config.learning_rate, wd=config.wd, beta=config.beta)
    elif config.opt == 'dynamic':
        optimizer = nigt.Dynamic(model.parameters(), lr=config.learning_rate, wd=config.wd, beta=config.beta, implicit=config.implicit)
    elif config.opt == 'dynamic_reg':
        optimizer = nigt.Dynamic_reg(model.parameters(), lr=config.learning_rate, wd=config.wd, beta=config.beta, implicit=config.implicit)
    elif config.opt == 'dynamic_reg_reset':
        optimizer = nigt.Dynamic_reg_reset(model.parameters(), lr=config.learning_rate, wd=config.wd, beta=config.beta, implicit=config.implicit, adaptive=config.adaptive)
    elif config.opt == 'random':
        optimizer = nigt.RandomSGDM(model.parameters(), lr=config.learning_rate, wd=config.wd, beta=config.beta, scale_type=config.scale_type)
    elif config.opt == 'randomol':
        optimizer = nigt.RandomOL(model.parameters(), lr=config.learning_rate, wd=config.wd, scale_type=config.scale_type, ol=config.ol, beta=config.beta ,beta2=config.beta2)
    elif config.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.beta, dampening=config.beta)
    losses = []

    iterations = 0

    for epoch in range(config.epochs):
        # if config.args.dataset == 'wikitext':
        #     train_ds = ds_loader.load_wikitext(config, tokenizer, split='train')
        # elif config.args.dataset == 'c4':
        #     train_ds = ds_loader.load_c4(config, tokenizer, split='train')
        # ds_path = f"data/manual_saves/wikitext-2-v1/bs-{config.batch_size}-ml-{model.config.context_length}-train"
        # try:
        #     wikitext = load_from_disk(ds_path)
        #     print("loaded!")
        # except FileNotFoundError:   
        #     print("file not found, generating from scratch")
        #     wikitext = load_dataset('wikitext', 'wikitext-2-v1', split='train')
        #     wikitext = wikitext.filter(lambda x: len(x['text']) > 1)
        #     wikitext.shuffle()
        #     wikitext = wikitext.map(lambda examples: tokenizer(examples["text"], 
        #                                                     padding=True,
        #                                                     truncation=True,
        #                                                     max_length=model.config.context_length),
        #                             remove_columns=["text"],
        #                             batched=True,
        #                             batch_size=config.batch_size)
        #     wikitext.save_to_disk(ds_path)
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
        # loader = DataLoader(train_ds.with_format('torch'), batch_size=config.batch_size)
        train_loader = load_train_data(config, tokenizer)
        test_loader = load_test_data(config, tokenizer)
        # train_iter = WikiText2(root='data/', split='train')
        # loader = DataLoader(train_iter, batch_size=config.batch_size, shuffle=True)
        pbar = tqdm(enumerate(train_loader))#, total=len(loader))
        running_loss = 0.0
        running_accuracy = 0.0
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
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
            epoch_train_loss += (loss.item() - epoch_train_loss)/(t+1.0)
            running_accuracy += (accuracy.item() - running_accuracy)/min(t+1.0, 1000.0)
            epoch_train_accuracy += (accuracy.item() - epoch_train_accuracy)/(t+1.0)

            # losses.append(loss.item())
            iterations += 1
            if iterations % 100 == 0:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy.item(),
                    "it_per_second": 1.0/delta_time
                },
                step = iterations)
            pbar.set_description(f"train epoch {epoch+1} iter {t}: train loss {loss.item():.5f}, running loss {running_loss:0.5f}, running accuracy {running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")

            for _ in range(config.ministeps-1):
                model.zero_grad()
                features, loss, accuracy = model(idx, mask)
                loss.backward()
                optimizer.step()

            if iterations % config.args.valid_frequency == 0:
                wandb.log({
                    "train/valid_freq_loss": epoch_train_loss,
                    "train/valid_freq_accuracy": epoch_train_accuracy
                },
                step = iterations)
                print("testing:")
                if config.opt == 'nigt' and config.recenter:
                    optimizer.x_to_w_()
                test(model, config, device, epoch, iterations, test_loader)
                if config.opt == 'nigt' and config.recenter:
                    optimizer.w_to_x_()
        wandb.log({
            "train/epoch_loss": epoch_train_loss,
            "train/epoch_accuracy": epoch_train_accuracy
        },
        step = iterations)
        print("testing:")
        if config.opt == 'nigt' and config.recenter:
            optimizer.x_to_w_()
        test(model, config, device, epoch, iterations)
        if config.opt == 'nigt' and config.recenter:
            optimizer.w_to_x_()

    return losses

def load_test_data(config, tokenizer):
    if config.args.dataset == 'wikitext':
        test_dataset = ds_loader.load_wikitext(config, tokenizer, split='test')
    elif config.args.dataset == 'c4':
        test_dataset = ds_loader.load_c4(config, tokenizer, split='validation')
    loader = DataLoader(test_dataset.with_format('torch'), batch_size=config.batch_size)
    return enumerate(loader)

def load_train_data(config, tokenizer):
    if config.args.dataset == 'wikitext':
        train_dataset = ds_loader.load_wikitext(config, tokenizer, split='train')
    elif config.args.dataset == 'c4':
        train_dataset = ds_loader.load_c4(config, tokenizer, split='train')
    loader = DataLoader(train_dataset.with_format('torch'), batch_size=config.batch_size)
    return enumerate(loader)


def test(model, config, device, epoch, iterations, loader=None):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    losses = []

    # if config.args.dataset == 'wikitext':
    #     test_dataset = ds_loader.load_wikitext(config, tokenizer, split='test')
    # elif config.args.dataset == 'c4':
    #     test_dataset = ds_loader.load_c4(config, tokenizer, split='validation')


    # ds_path = f"data/manual_saves/wikitext-2-v1/bs-{config.batch_size}-ml-{model.config.context_length}-test"
    # try:
    #     wikitext = load_from_disk(ds_path)
    #     print("loaded!")
    # except FileNotFoundError:   
    #     print("file not found, generating from scratch")
    #     wikitext = load_dataset('wikitext', 'wikitext-2-v1', split='test')
    #     wikitext = wikitext.filter(lambda x: len(x['text']) > 1)
    #     wikitext.shuffle()
    #     wikitext = wikitext.map(lambda examples: tokenizer(examples["text"], 
    #                                                     padding=True,
    #                                                     truncation=True,
    #                                                     max_length=model.config.context_length),
    #                             remove_columns=["text"],
    #                             batched=True,
    #                             batch_size=config.batch_size)
    #     wikitext.save_to_disk(ds_path)
    if loader is None:
        loader = load_test_data(config, tokenizer)#DataLoader(test_dataset.with_format('torch'), batch_size=config.batch_size)

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
            "test/loss": running_loss,
            "test/accuracy": running_accuracy
        },
        step=iterations)
        pbar.set_description(f"test epoch {epoch+1} iter {t}: train loss {loss.item():.5f}, running loss {running_loss:0.5f}, running accuracy {running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")
        if 

    return losses



def initialize_and_train_model(args):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=10,
        embedding_dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        args=args,
        use_diag=args.use_diag)
    print(args.use_diag)

    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,#*args.batch_size/args.ministeps,
        epochs=args.epochs,
        wd=args.weight_decay,
        opt=args.opt,
        ministeps=args.ministeps,
        eps=args.eps,
        recenter=(args.recenter == 'true') or (args.recenter == True),
        beta=args.beta,
        implicit=(args.implicit == 'true') or (args.implicit == True),
        adaptive=(args.adaptive == 'true') or (args.adaptive == True),
        scale_type=args.scale_type,
        beta2=args.beta2,
        ol=args.ol,
        args=args,
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



    

    


    
