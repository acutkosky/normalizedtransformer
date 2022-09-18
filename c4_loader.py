from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange
import random


def shift_labels(batch):
    batch["labels"] = F.pad(batch["labels"][:, 1:], (0, 1), value=-100)
    return batch

def split_sequences(batch, max_length):
    # assumes that batch has shape [B, L] where L is a multiple of max_length
    batch["size"] = batch["input_ids"].size()
    batch["input_ids"] = rearrange(batch["input_ids"], 'b (l c) -> (b l) c')
    batch["labels"] = rearrange(batch["labels"], 'b (l c) -> (b l) c')
    return batch

def postprocess_collate_fn(collate_fn, post_fn):
    old_torch_call = collate_fn.torch_call
    def new_torch_call(self, *args, **kwargs):
        batch = old_torch_call(self, *args, **kwargs)
        return post_fn(batch)
    collate_fn.torch_call = new_torch_call
    return collate_fn




def get_c4_loader_next_token(tokenizer, split, batch_size, max_length=None, shuffle_buffer_size=0,
                             pad_to_multiple_of=None, mlm=False, mlm_probability=0, random_start=False, 
                             num_workers=2, **collator_args):
    collate_fn = DataCollatorForLanguageModeling(tokenizer,
                                                 mlm=mlm,
                                                 mlm_probability=mlm_probability,
                                                 pad_to_multiple_of=pad_to_multiple_of,
                                                 **collator_args)
    collate_fn = postprocess_collate_fn(collate_fn, shift_labels)
    return get_c4_loader_from_collate_fn(tokenizer=tokenizer,
                                         split=split,
                                         batch_size=batch_size,
                                         shuffle_buffer_size=shuffle_buffer_size,
                                         max_length=max_length,
                                         random_start=random_start,
                                         collate_fn=collate_fn,
                                         num_workers=num_workers)
                                    

def get_c4_loader_lm(tokenizer, split, batch_size, mlm, mlm_probability, shuffle_buffer_size=0,
                     max_length=None, pad_to_multiple_of=0, random_start=False,
                     num_workers=2, **collator_args):
    collate_fn = DataCollatorForLanguageModeling(tokenizer,
                                                 mlm=mlm,
                                                 mlm_probability=mlm_probability,
                                                 pad_to_multiple_of=pad_to_multiple_of,
                                                 **collator_args)

    return get_c4_loader_from_collate_fn(tokenizer=tokenizer,
                                         split=split,
                                         batch_size=batch_size,
                                         max_length=max_length,
                                         shuffle_buffer_size=shuffle_buffer_size,
                                         random_start=random_start,
                                         collate_fn=collate_fn,
                                         num_workers=num_workers)


def get_c4_loader_from_collate_fn(tokenizer, split, batch_size, max_length, shuffle_buffer_size, random_start, collate_fn, num_workers=2):
    ds_path = f"/projectnb/aclab/datasets/c4/en/"
    c4 = load_dataset('c4', 'en', data_dir=ds_path, streaming=True, split=split)
    c4 = c4.filter(lambda x: len(x['text']) > 1)
    if shuffle_buffer_size > 0:
        c4 = c4.shuffle(buffer_size=shuffle_buffer_size)
    if random_start:
        c4 = c4.map(lambda examples: {"text": examples["text"][random.randint(0,len(examples["text"])):]})
    c4 = c4.map(lambda examples: tokenizer(examples["text"], 
                                            padding=True,
                                            truncation=True,
                                            max_length=max_length),
                                remove_columns=["text", "timestamp", "url"],
                                batched=True,
                                batch_size=num_workers//2)
    c4 = c4.with_format("torch")


    dataloader = DataLoader(c4,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            num_workers=num_workers//2)
    return dataloader

