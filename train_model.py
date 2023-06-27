import os
import glob
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def main():
    # Step 1: Prepare the data
    data_folder = "./input"
    combined_file = "./output/combined.txt"

    # Combine the text files into one
    with open(combined_file, "w", encoding="utf-8") as outfile:
        for file_path in glob.glob(os.path.join(data_folder, "*.txt")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                outfile.write(infile.read() + "\n")

    # Step 2: Install the required libraries (assuming Transformers is already installed)

    # Step 3: Tokenize the data
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokenized_data = []

    # Step 4: Prepare the training dataset
    sequence_length = 128
    training_examples = []

    with open(combined_file, "r", encoding="utf-8", errors="ignore") as file:
        training_text = file.read()
        stripped_training_text = training_text.replace('!', '').replace('_', '').replace('\n', ' ')
        stripped_training_text = stripped_training_text.split()
        for i in range(0, len(stripped_training_text), sequence_length):
            example = ' '.join(stripped_training_text[i:i + sequence_length])
            training_examples.append(example)

    tokenized_examples = []
    for example in training_examples:
        tokenized_example = tokenizer.encode(example)
        tokenized_examples.append(torch.tensor(tokenized_example))

    training_tensors = pad_sequence(tokenized_examples, batch_first=True)

    # Step 5: Create a GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Step 6: Train the model
    output_dir = "./model"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=500
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=training_tensors
    )

    trainer.train()

    trainer.save_model(output_dir)

