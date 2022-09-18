
import torch
from optional_module import NoOpModule
import c4_loader
from omegaconf import OmegaConf
import argparse
from tqdm import tqdm
from model import StackedAttention, ModelConfig
import online_opt
import transformers

def multiresample_test(model, optimizer, load_generator, device, samples, logger=NoOpModule(), use_tqdm=False):
    total_cur_loss = 0
    total_difference = 0
    total_average = 0
    total_std_dev = 0 
    if use_tqdm:
        generator = tqdm(range(samples))
    else:
        generator = range(samples)
    for _ in generator:
        cur_loss, difference, average, std_dev = run_resample_test(model, optimizer, load_generator, device, 3, use_tqdm=False)
        total_cur_loss += cur_loss
        total_difference += difference
        total_average += average
        total_std_dev += std_dev**2
    
    cur_loss = total_cur_loss / samples
    average = total_average / samples
    difference = total_difference / samples
    std_dev = torch.sqrt(total_std_dev / samples)

    return cur_loss, difference, average, std_dev





def resample_test(optimizer, closure, samples, logger=NoOpModule(), use_tqdm=False):
    optimizer.swap_prev_state()

    features, prev_loss, accuracy = closure()

    optimizer.swap_prev_state()
    optimizer.swapstate()

    features, cur_loss, accuracy = closure()

    optimizer.swapstate

    difference = prev_loss - cur_loss

    total_product = 0
    generator = range(samples)
    variance = 0
    if use_tqdm:
        generator = tqdm(generator)
    for s in generator:
        optimizer.resample()
        closure()
        inner_product, _ = optimizer.get_inner_product()
        total_product += inner_product
        variance += inner_product**2
    
    average = total_product / samples
    variance = variance/samples - average**2



    logger.log({
        'verification/difference': difference,
        'verification/average_product': average,
        'verification/difference_minus_average': difference-average,
        'verification/difference_average_relative_error': torch.abs(difference-average)/(torch.abs(difference) + torch.abs(average))
    },
    commit=False)

    return cur_loss, difference, average, torch.sqrt(variance)




# parser = argparse.ArgumentParser(description='Simple pretraining thing')
# parser.add_argument('--config', type=str, default='configs/conf.yaml')
# args = parser.parse_args()

def runtest(conf='configs/confadam.yaml'):
    config = OmegaConf.load(conf)


    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    loader = c4_loader.get_c4_loader_next_token(tokenizer,
                        split='train',
                        batch_size=config.batch_size,
                        max_length=config.context_length,
                        pad_to_multiple_of=config.context_length,
                        num_workers=2)
        
    load_generator = enumerate(loader)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=10,
        embedding_dim=config.dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        args=config,
        use_diag=config.use_diag)

    device = 'cpu'
    if torch.cuda.is_available():            
        device = torch.cuda.current_device()

    model = StackedAttention(model_config).to(device)

    optimizer = online_opt.RandomOL(model.parameters(), lr=config.lr, wd=config.wd, scale_type=config.scale_type, ol=config.ol, beta=config.beta ,beta2=config.beta2, beta3=config.beta3)

    # do some steps just to get the ball rolling:

    burnin = 1000

    iterate = 0

    print("burning in!")
    for (t, example) in tqdm(load_generator, total=burnin):

        iterate += 1
        # print(strings)
        # encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
        idx = example['input_ids'].to(device)
        mask = example['attention_mask'].to(device)
        labels = example['labels'].to(device)
        # print(idx.size(),labels.size())


        model.zero_grad()
        features, loss, accuracy = model(idx, mask, labels)
        loss.backward()

        optimizer.step()
        if iterate > burnin:
            break

    print("finished burnin")
    (t, example) = load_generator.__next__()


    idx = example['input_ids'].to(device)
    mask = example['attention_mask'].to(device)



    # def closure():
    #     model.zero_grad()
    #     features, loss, accuracy = model(idx, mask)
    #     loss.backward()
    #     return features, loss, accuracy

    cur_loss, difference, average, std_dev = run_resample_test(model, optimizer, load_generator, device, samples=1000, use_tqdm=True )#resample_test(optimizer, closure, samples=1000, use_tqdm=True)

    print(f"current_loss: {cur_loss}\ndifference: {difference}\naverage: {average}\nstd dev: {std_dev}")

    return model, optimizer, load_generator, device

def run_resample_test(model, optimizer, load_generator, device, samples, use_tqdm=False):
    (t, example) = load_generator.__next__()
    idx = example['input_ids'].to(device)
    mask = example['attention_mask'].to(device)
    labels = example['labels'].to(device)


    def closure():
        model.zero_grad()
        features, loss, accuracy = model(idx, mask, labels)
        loss.backward()
        return features, loss, accuracy

    return resample_test(optimizer, closure, samples=samples, use_tqdm=use_tqdm)


      



if __name__ == '__main__':
    runtest()