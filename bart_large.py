#%%
!pip install transformers
!pip install datasets
!pip install evaluate
#%%
import pandas as pd
import random
import re
from datasets import Dataset
from evaluate import load
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import torch

#%%
pip install evaluate
#%%
!pip install jiwer
#%%
import requests

base_url = "https://raw.githubusercontent.com/google-research-datasets/wiki-split/master/"
files = ["tune.tsv", "validation.tsv", "test.tsv"]

for file in files:
    url = base_url + file
    response = requests.get(url)
    if response.status_code == 200:
        with open(file, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {file}")
    else:
        print(f"Failed to download {file}")

#%%
# Load datasets
train_df = pd.read_csv("tune.tsv", sep='\t', header=None, names=['original', 'split'])
val_df = pd.read_csv("validation.tsv", sep='\t', header=None, names=['original', 'split'])
test_df = pd.read_csv("test.tsv", sep='\t', header=None, names=['original', 'split'])
#%%
sentence = train_df.head(1)

print(sentence.to_string())
#%%
# Keep only the original sentence column
train_sentences = train_df['original'].dropna().tolist()
val_sentences = val_df['original'].dropna().tolist()
test_sentences = test_df['original'].dropna().tolist()

#%%
# prompt: WHAT IS LENGTH OF EACH dataset

print(f"Length of train dataset: {len(train_sentences)}")
print(f"Length of validation dataset: {len(val_sentences)}")
print(f"Length of test dataset: {len(test_sentences)}")

#%%
# prompt: print first item in train_sentences completely

train_sentences[0]

#%%
import random
import pandas as pd

# Map of keyboard neighbors for typo simulation
keyboard_adj = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx',
    'e': 'wsdr', 'f': 'rtgdvc', 'g': 'tyfhvb', 'h': 'yugjnb',
    'i': 'ujko', 'j': 'uikhmn', 'k': 'ijolm', 'l': 'kop',
    'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
    'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
    'y': 'tghu', 'z': 'asx',
}

def introduce_typos(sentence, typo_prob=0.2):
    def typo(word):
        if (
            len(word) <= 3 or
            not word.isalpha() or
            word[0].isupper() or
            (len(word) > 1 and word[1].isupper()) or
            random.random() > typo_prob
        ):
            return word

        ops = ['delete', 'swap', 'replace', 'add', 'keyboard']
        op = random.choice(ops)
        i = random.randint(0, len(word) - 1)
        c = word[i].lower()

        if op == 'delete':
            return word[:i] + word[i+1:]
        elif op == 'swap' and i < len(word) - 1:
            return word[:i] + word[i+1] + word[i] + word[i+2:]
        elif op == 'replace':
            return word[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[i+1:]
        elif op == 'add':
            return word[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[i:]
        elif op == 'keyboard' and c in keyboard_adj:
            replacement = random.choice(keyboard_adj[c])
            return word[:i] + replacement + word[i+1:]
        return word

    return ' '.join([typo(w) for w in sentence.split()])

#%%
def generate_dataset(sentences, typo_prob=0.2, clean_percent=0.15, n_augmented=2):
    corrupted = []
    targets = []

    for sent in sentences:
        if random.random() < clean_percent:
            corrupted.append(sent)
            targets.append(sent)
        else:
            for _ in range(n_augmented):
                corrupted.append(introduce_typos(sent, typo_prob))
                targets.append(sent)

    return pd.DataFrame({'input': corrupted, 'target': targets})


train_data = generate_dataset(train_sentences, typo_prob=0.2, clean_percent=0.15)
val_data = generate_dataset(val_sentences, typo_prob=0.2, clean_percent=0.15)
test_data = generate_dataset(test_sentences, typo_prob=0.2, clean_percent=0.15)

#%%
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

#%%
train_dataset[0] , val_dataset[0], test_dataset[0]
#%%

print(f"Length of train dataset: {len(train_dataset)}")
print(f"Length of validation dataset: {len(val_dataset)}")
print(f"Length of test dataset: {len(test_dataset)}")

#%%
# prompt: keep first 1000 rows in test_dataset

test_dataset = test_dataset.select(range(1000))
print(f"Length of test dataset: {len(test_dataset)}")

#%%
# prompt: get the max length of the train and val dataset inputs

max_train_len = 0
for example in train_dataset:
    max_train_len = max(max_train_len, len(example['input']), len(example['target']))

max_val_len = 0
for example in val_dataset:
    max_val_len = max(max_val_len, len(example['input']), len(example['target']))

print(f"Max train length: {max_train_len}")
print(f"Max val length: {max_val_len}")


max_length = max(max_train_len, max_val_len)  # = 346

#%%
def preprocess(tokenizer, dataset):
    def tokenize(example):
        model_inputs = tokenizer(example["input"], max_length=128, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example["target"], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(tokenize, batched=True)

#%%
import os
def train_model(model_name, tokenizer_cls, model_cls):

    tokenizer = tokenizer_cls.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name)

    tokenized_train = preprocess(tokenizer, train_dataset)
    tokenized_val = preprocess(tokenizer, val_dataset)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{model_name}-finetuned-spell",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_dir="./logs",
        fp16=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    try:
        model = model_cls.from_pretrained(f"./{model_name}-finetuned-spell")
        tokenizer = tokenizer_cls.from_pretrained(f"./{model_name}-finetuned-spell")
        # Load trainer state directly on instantiation.
        trainer = Seq2SeqTrainer(
            model=model,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
        )
        print("Loading saved model...")
        trainer.train(resume_from_checkpoint=os.path.join(f"./{model_name}-finetuned-spell/checkpoint-3750"))

        print("Returning")
        return trainer, tokenizer, model
    except Exception as e:
        print(f"Error loading saved model: {e}")
        pass # or handle the exception appropriately
    print("Training model...")
    trainer.train()

    # save the model, tokenizer and trainer
    model.save_pretrained(f"./{model_name}-finetuned-spell")
    tokenizer.save_pretrained(f"./{model_name}-finetuned-spell")
    trainer.save_state()


    return trainer, tokenizer, model

#%%
bart_trainer, bart_tokenizer, bart_model = train_model("facebook/bart-large", BartTokenizer, BartForConditionalGeneration)

#%%
pip install python-Levenshtein
#%%
import Levenshtein

def compute_levenshtein(predictions, references):
    distances = []
    for pred, ref in zip(predictions, references):
        dist = Levenshtein.distance(pred, ref)
        distances.append(dist)
    avg_distance = sum(distances) / len(distances)
    return {"avg_levenshtein_distance": avg_distance}

#%%
from evaluate import load
import Levenshtein

cer = load("cer")
wer = load("wer")

def evaluate_model_metrics_batched(trainer, tokenizer, dataset, batch_size=32):
    inputs = dataset['input']
    targets = dataset['target']

    preds = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        tokenized = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(trainer.model.device)
        output_ids = trainer.model.generate(**tokenized, max_length=128)
        batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds.extend(batch_preds)

    cer_score = cer.compute(predictions=preds, references=targets)
    wer_score = wer.compute(predictions=preds, references=targets)
    levenshtein_score = compute_levenshtein(preds, targets)

    return {
        "CER": cer_score,
        "WER": wer_score,
        "Levenshtein": levenshtein_score
    }, preds


bart_metrics, bart_preds = evaluate_model_metrics_batched(bart_trainer, bart_tokenizer, test_dataset)

#%%
bart_metrics
#%%
def correct_sentence(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


user_input = "Ths is a sentnce with spleling errors."
print("User Input:", user_input)
print("BART Corrected:", correct_sentence(bart_model, bart_tokenizer, user_input))

#%%
user_input = "I haate Inter Milan fery much"
print("User Input:", user_input)
print("BART Corrected:", correct_sentence(bart_model, bart_tokenizer, user_input))
#%%
examples = [
    "I havw a dream about technology.",
    "The quick brown fox jumped over the lazey dog.",
    "Pleas correct this entire sentence with many typos."
]

for s in examples:
    print(f"\nOriginal:  {s}")
    print(f"Corrected: {correct_sentence(bart_model, bart_tokenizer, s)}")

#%%
for i in range(5):
    idx = random.randint(0, len(test_dataset))
    print(f"\nOriginal:  {test_dataset[idx]['input']}")
    print(f"Corrected: {correct_sentence(bart_model, bart_tokenizer, test_dataset[idx]['input'])}")
#%%
original_grammer = "I have 3 shrits"
corrected_sentence = correct_sentence(bart_model, bart_tokenizer, original_grammer)
print(f"Original:  {original_grammer}")
print(f"Corrected: {corrected_sentence}")
#%%
!pip install rapidfuzz

from rapidfuzz import fuzz

def fuzzy_ratio_test(dataset):

  ratios = []
  for example in dataset:
      ratio = fuzz.ratio(example['input'], example['target'])
      ratios.append(ratio)
  return ratios

fuzzy_ratios = fuzzy_ratio_test(test_dataset)

print(fuzzy_ratios[:10])