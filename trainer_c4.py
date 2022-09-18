

from torchtext.datasets import WikiText2
import transformers
from transformers import DataCollatorForLanguageModeling
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
import online_opt
from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf
from optional_module import optional_module
import ds_loader
import c4_loader
import wandb

parser = argparse.ArgumentParser(description='Simple pretraining thing')

# parser.add_argument('--logdir', default='./runs/', help='Tensorboard logdir')
# parser.add_argument('--save_path', default='awesomeplot.png', help='where to save data')
# parser.add_argument('--learning_rate', '-lr', type=float, default=6e-4)
# parser.add_argument('--beta', type=float, default=0.99)
# parser.add_argument('--beta2', type=float, default=0.99)
# parser.add_argument('--dim', type=int, default=512)
# parser.add_argument('--eps', '-eps', type=float, default=6e-4)
# parser.add_argument('--n_layers', type=int, default=12)
# parser.add_argument('--n_heads', type=int, default=1)
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--epochs', type=int, default=10)
# parser.add_argument('--ministeps', type=int, default=1)
# parser.add_argument('--weight_decay', '-wd', type=float, default=0.0)
# parser.add_argument('--use_diag', '-diag', type=str, default='true')
# parser.add_argument('--scale_type', type=str, default='random')
# parser.add_argument('--recenter', type=str, default='true')
# parser.add_argument('--implicit', type=str, default='true')
# parser.add_argument('--ol', type=str, default='sgd', choices=['ogd', 'dynamicpf', 'optogd', 'optadam'])
# parser.add_argument('--adaptive', type=str, default='true')
parser.add_argument('--config', type=str, default='configs/conf.yaml')
# parser.add_argument('--opt', '-opt', type=str, default='adamw', choices=['adamw', 'perpopt', 'expmd',
                                    # 'expmdnorm', 'nigt', 'dynamic', 'nigt_lamb', 'dynamic_reg', 'dynamic_reg_reset',
                                    # 'random','sgd', 'randomol'])


# args = parser.parse_args()
# args = OmegaConf.create(vars(args))
# if args.config is not None:
#     conf = OmegaConf.load(args.config)
#     args = OmegaConf.merge(args, conf)

# print(args)
# args_dict = OmegaConf.to_container(args)

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
    print(config)
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    if config.opt == 'adamw' or config.opt == 'adamw_warmup':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd, betas=(config.beta, config.beta2))
    elif config.opt == 'perpopt':
        optimizer = perpopt.PerpOpt(model.parameters(), lr=config.lr, wd=config.wd, beta=config.beta, alignbeta=1.0)
    elif config.opt == 'expmd':
        optimizer = expmd.ExpMD(model.parameters(), lr=config.lr, wd=config.wd, eps=config.eps)
    elif config.opt == 'expmdnorm':
        optimizer = expmd.ExpMDNorm(model.parameters(), lr=config.lr, wd=config.wd, eps=config.eps)
    elif config.opt == 'nigt':
        optimizer = nigt.Nigt(model.parameters(), lr=config.lr, wd=config.wd, beta=config.beta)
    elif config.opt == 'nigt_lamb':
        optimizer = nigt.Nigt(model.parameters(), lr=config.lr, wd=config.wd, beta=config.beta)
    elif config.opt == 'dynamic':
        optimizer = nigt.Dynamic(model.parameters(), lr=config.lr, wd=config.wd, beta=config.beta, implicit=config.implicit)
    elif config.opt == 'dynamic_reg':
        optimizer = nigt.Dynamic_reg(model.parameters(), lr=config.lr, wd=config.wd, beta=config.beta, implicit=config.implicit)
    elif config.opt == 'dynamic_reg_reset':
        optimizer = nigt.Dynamic_reg_reset(model.parameters(), lr=config.lr, wd=config.wd, beta=config.beta, implicit=config.implicit, adaptive=config.adaptive)
    elif config.opt == 'random':
        optimizer = nigt.RandomSGDM(model.parameters(), lr=config.lr, wd=config.wd, beta=config.beta, scale_type=config.scale_type)
    elif config.opt == 'randomol':
        optimizer = online_opt.RandomOL(model.parameters(), lr=config.lr, wd=config.wd, scale_type=config.scale_type, ol=config.ol, beta=config.beta ,beta2=config.beta2, beta3=config.beta3, logger=wandb)
    elif config.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.beta, dampening=config.beta)
    losses = []

    iterations = 0
    train_loader = load_train_data(config, tokenizer)
    train_iter = enumerate(train_loader)
    total = config.valid_frequency/config.batch_size
    if config.dataset == 'c4':
        test_loader = load_test_data(config, tokenizer)
        test_iter = enumerate(test_loader)
    else:
        test_loader = None
        test_iter = None
    tokens = 0
    total_difference = 0
    for epoch in range(config.epochs):
        if config.dataset == 'wikitext':
            train_loader = load_train_data(config, tokenizer)
            train_iter = enumerate(train_loader)
            total = len(train_iter)
        pbar = tqdm(train_iter, total=total)#, total=len(loader))
        running_loss = 0.0
        running_accuracy = 0.0
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        last_time = time.monotonic()
        cur_run_it = 0
        for t, strings in pbar:
            cur_run_it += 1
            # print(strings)
            # encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
            idx = strings['input_ids'].to(device)
            mask = strings['attention_mask'].to(device)
            labels = strings['labels'].to(device)


            def closure():
                model.zero_grad()
                features, loss, accuracy = model(idx, mask, labels)
                loss.backward()
                return features, loss, accuracy

            def inference_closure():
                with torch.no_grad():
                    features, loss, accuracy = model(idx, mask, labels)
                return features, loss, accuracy

            if config.log_differences:
                optimizer.swap_prev_state()
                features, prev_loss, accuracy = inference_closure()
                optimizer.swap_prev_state() 


                optimizer.swapstate()
                features, cur_loss, accuracy = inference_closure()
                optimizer.swapstate()

                difference = prev_loss - cur_loss
                total_difference += difference

            features, loss, accuracy = closure()

            optimizer.step()
            tokens += (mask >= 0).sum()


            current_time = time.monotonic()
            delta_time = current_time - last_time
            last_time = current_time
            running_loss += (loss.item() - running_loss)/min(cur_run_it, 1000.0)
            epoch_train_loss += (loss.item() - epoch_train_loss)/(cur_run_it)
            running_accuracy += (accuracy.item() - running_accuracy)/min(cur_run_it, 1000.0)
            epoch_train_accuracy += (accuracy.item() - epoch_train_accuracy)/(cur_run_it)

            # losses.append(loss.item())
            iterations += 1

            if config.opt == 'adamw_warmup':
                lr = config.lr * min(1, float(iterations) / float(max(1, config.warmup_steps)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                if iterations % 100 == 0:
                    wandb.log({
                        'optimizer/learned_lr': lr
                    },
                    step = iterations)


            if iterations % 100 == 0:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy.item(),
                    "it_per_second": 1.0/delta_time,
                }
                if config.log_differences:
                    log_dict["optimizer/loss_difference"] = difference
                    log_dict["optimizer/total_loss_difference"] = total_difference
                wandb.log(log_dict, step = iterations)
            pbar.set_description(f"train epoch {epoch+1} iter {t}: current loss {loss.item():.5f}, running loss {running_loss:0.5f}, running accuracy {running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")

            for _ in range(config.ministeps-1):
                model.zero_grad()
                features, loss, accuracy = model(idx, mask)
                loss.backward()
                optimizer.step()
            if (t+1) % total == 0:
                break

        wandb.log({
            "train/epoch_loss": epoch_train_loss,
            "train/epoch_accuracy": epoch_train_accuracy
        },
        step = iterations)
        print("testing:")
        if config.opt == 'nigt' and config.recenter:
            optimizer.x_to_w_()
        if config.opt == 'randomol' and config.swapol:
            optimizer.swapstate()
        _, finished = test(model, config, device, epoch, iterations, test_iter)
        if config.opt == 'randomol' and config.swapol:
            optimizer.swapstate()
        if not finished and conf.dataset == 'c4':
            test_loader = load_test_data(config, tokenizer)
            test_iter = enumerate(test_loader)
        if config.opt == 'nigt' and config.recenter:
            optimizer.w_to_x_()

    return losses

def load_test_data(config, tokenizer):
    if config.dataset == 'wikitext':
        loader = DataLoader(ds_loader.load_wikitext(config, tokenizer, split='test').with_format('torch'), batch_size=config.batch_size)
    elif config.dataset == 'c4':
        loader = c4_loader.get_c4_loader_next_token(tokenizer,
                    split='validation',
                    batch_size=config.batch_size,
                    max_length=config.context_length,
                    pad_to_multiple_of=config.context_length)
    return loader

def load_train_data(config, tokenizer):
    if config.dataset == 'wikitext':
        loader = DataLoader(ds_loader.load_wikitext(config, tokenizer, split='train'), batch_size=config.batch_size)
    elif config.dataset == 'c4':
        loader = c4_loader.get_c4_loader_next_token(tokenizer,
                    split='train',
                    batch_size=config.batch_size,
                    max_length=config.context_length,
                    pad_to_multiple_of=config.context_length,
                    num_workers=2)
    return loader


def test(model, config, device, epoch, iterations, test_iter=None):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    losses = []

    if test_iter is None:
        loader = load_test_data(config, tokenizer)#DataLoader(test_dataset.with_format('torch'), batch_size=config.batch_size)
        test_iter = enumerate(loader)
        total = len(test_iter)
    else:
        total = config.valid_size / config.batch_size

    # train_iter = WikiText2(root='data/', split='train')
    # loader = DataLoader(train_iter, batch_size=config.batch_size, shuffle=True)
    pbar = tqdm(test_iter, total=total)#, total=len(loader))
    running_loss = 0.0
    running_accuracy = 0.0
    last_time = time.monotonic()
    cur_run_it = 0
    finished = False
    for t, strings in pbar:
        # encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
        idx = strings['input_ids'].to(device)
        mask = strings['attention_mask'].to(device)
        labels = strings['labels'].to(device)
        cur_run_it += 1
        
        with torch.no_grad():
            features, loss, accuracy = model(idx, mask, labels)


        current_time = time.monotonic()
        delta_time = current_time - last_time
        last_time = current_time
        running_loss += (loss.item() - running_loss)/(cur_run_it)
        running_accuracy += (accuracy.item() - running_accuracy)/(cur_run_it)

        losses.append(loss.item())
        wandb.log({
            "test/loss": running_loss,
            "test/accuracy": running_accuracy
        },
        step=iterations)
        pbar.set_description(f"test epoch {epoch+1} iter {t}: current loss {loss.item():.5f}, running loss {running_loss:0.5f}, running accuracy {running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")
        if (t+1) % total == 0:
            finished = True
            break

    return losses, finished



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
        learning_rate=args.lr,#*args.batch_size/args.ministeps,
        epochs=args.epochs,
        wd=args.wd,
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

    losses = train(attention_model, args, device)

    # plt.plot(losses)
    # plt.xlabel('iterations')
    # plt.ylabel('loss')
    # plt.title('Loss of awesome model!')

    # plt.savefig(args.save_path)

    

if __name__=='__main__':
    args = parser.parse_args()
    args = OmegaConf.create(vars(args))
    if args.config is not None:
        conf = OmegaConf.load(args.config)
        args = OmegaConf.merge(args, conf)
    wandb = optional_module(wandb, args.logging)
    wandb.init()
    wandb.config.update(args)
    initialize_and_train_model(args)



    

    


    
