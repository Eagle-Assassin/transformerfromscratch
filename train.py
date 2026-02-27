from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from  datasets import load_dataset
from dataset import OscarNavDataset,encode,causal_mask

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

import re
import warnings
from tqdm import tqdm
from collections import Counter
from pathlib import Path
import os



dataset =load_dataset("OscarNav/spa-eng")

def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx = encode("[SOS]", tokenizer_tgt)[0]
    eos_idx = encode("[EOS]", tokenizer_tgt)[0]

    #Precompute the encoder output nad reuse it for every step
    encoder_output= model.encode(source,source_mask)

    #Initialize the decoder input with sos token
    
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1)== max_len:
            break

        #build mask for target
        decoder_mask=causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #Calculate Output
        out = model.decode(encoder_output,source_mask,decoder_input,decoder_mask)

        #get next token
        prob = model.project(out[:,-1])
        _,next_word = torch.max(prob,dim=1)
        next_word_id = next_word.item()
        next_word_tensor = torch.tensor([[next_word_id]], dtype=torch.long).to(device)
        decoder_input = torch.cat([decoder_input, next_word_tensor], dim=1)
        # decoder_input = torch.cat([decoder_input,torch.empty(1,1).type_as(source).fill_(next_word.item().to(device))],dim=1)

        if next_word ==eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_step,writer,num_examples=2):

    model.eval()
    count =0
    id_to_word = {v: k for k, v in tokenizer_tgt.items()}

    source_texts =[]
    expected =[]
    predicted =[]

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch["encoder_input"].to(device) #(b,seq_len)
            encoder_mask  = batch["encoder_mask"].to(device) #(b,1,1,seq_len)

            #check that the batch size is 1

            assert encoder_input.size(0)==1,"Batch size must be 1 for validation"


            model_out = greedy_decode( model, encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)

            source_text=batch["src_text"][0]
            target_text=batch["tgt_text"][0]
            model_out_text = decode(model_out.detach().cpu().numpy(), id_to_word)
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            #Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-"*console_width)
                break
        
    if writer:
        #Evaulate the character error rate
        #Compute  the char error rate
        mertic = torchmetrics.CharErrorRate()
        cer = mertic(predicted,expected)
        writer.add_scalar("validation cer", cer ,global_step)
        writer.flush()

        #Compute the word error rate
        metric= torchmetrics.WordErrorRate()
        wer= mertic(predicted,expected)
        writer.add_scalar('validation wer',wer,global_step)
        writer.flush()

        #Compute BLEU metric
        metric=torchmetrics.BLEUScore()
        bleu=metric(predicted,expected)
        writer.add_scalar('validation BLEU',bleu,global_step)
        writer.flush()


def standardize(text):
    text=text.lower()
    text=re.sub(r"[^\w\s]","",text) #remove puntuation
    return text

def tokenize(sentence):
    return sentence.split()

def decode(ids, id_to_word):
    words = []
    for i in ids:
        word = id_to_word.get(i, "[unk]")
        if word == "[EOS]":
            break
        if word not in ["[PAD]", "[SOS]"]:
            words.append(word)
    return " ".join(words)




def build_vocab(sentences, max_vocab_size=15000):
    counter = Counter()
    
    for sentence in sentences:
        counter.update(tokenize(sentence))
    
    special_tokens = ["[PAD]", "[unk]", "[SOS]", "[EOS]"]
    
    most_common = counter.most_common(max_vocab_size - len(special_tokens))
    
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    return vocab


def get_ds(config):
    dataset =load_dataset("OscarNav/spa-eng")

    #Use only the  first 50,000 setance pairs
    dataset=dataset['train'].select(range(50000))
    english_sentence=dataset[config['lang_src']]
    spanish_sentence=dataset[config['lang_tgt']]

    english_sentence=[standardize(s) for s in english_sentence]
    spanish_sentence=[standardize(s) for s in spanish_sentence]

    #Build Tokenizers
    src_vocab = build_vocab(english_sentence, 15000)
    tgt_vocab = build_vocab(spanish_sentence, 15000)

    #Keep 90% training and 10% validation
    train_ds_size= int(0.9* len(dataset))
    val_ds_size=len(dataset)-train_ds_size

    train_ds_raw,val_ds_raw =random_split(dataset,[train_ds_size,val_ds_size])

    train_ds = OscarNavDataset(train_ds_raw,src_vocab,tgt_vocab,config["lang_src"],config['lang_tgt'],config['seq_len'])
    val_ds = OscarNavDataset(val_ds_raw,src_vocab,tgt_vocab,config["lang_src"],config['lang_tgt'],config['seq_len'])

    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,src_vocab,tgt_vocab


def get_model(config,vocab_src_len,vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return (model)


def train_model(config):
    #Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else 'cpu'
    print("Using device:",device)

    if (device =="cuda"):
        print(f"Device Name:{torch.cuda.get_device_name(device.index)}")
        print(f"Device Memory:{torch.cuda.get_device_properties(device.index).total_memory/1024*3} GB")
    elif (device=='mps'):
        print(f"Device name: <mps")

    else:
        print("NOTE: If you have a GPU, consider using it for training.")

    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True) 

    train_loader,val_dataloader,tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config,len(tokenizer_src),len(tokenizer_tgt)).to(device)

    #Tensor Board
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    #If the user specified a model to preload beore training
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    
    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src["[PAD]"],label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_loader,desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b,seq_len)
            decoder_input = batch['decoder_input'].to(device) # (b,seq_len)
            encoder_mask  = batch['encoder_mask'].to(device) # (b,seq_len)
            decoder_mask  = batch['decoder_mask'].to(device) # (b,seq_len)

            #Run the Tensor through encoder Decoder and the projection Layer
            encoder_output = model.encode(encoder_input,encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            #Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            #Compute the loss using simple cross entropy
            loss =loss_fn(proj_output.view(-1,len(tokenizer_tgt)),label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

             # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            #back porpagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


