import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments
import torch



def main():

    # Set the GPU device index
    device_index = 0  # Change this to the desired GPU device index

    # Set the device
    torch.cuda.set_device(device_index)

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(device_index)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")

    # Set the paths for input data and the model directory
    input_folder = "./input/"
    output_dir = "./model/"

    # Load the pretrained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Preprocess the data
    file_paths = [os.path.join(input_folder, file_name) for file_name in os.listdir(input_folder)]

    # Read and concatenate the contents of all the files
    corpus = ""
    for file_path in file_paths:
        with open(file_path, "r", encoding="latin-1") as file:
            corpus += file.read()

    # Tokenize the corpus
    inputs = tokenizer.encode(corpus, return_tensors="pt")

    # Save the tokenized inputs to a file
    tokenized_file = "./input/tokenized.txt"
    with open(tokenized_file, "w", encoding="utf-8") as file:
        for input_ids in inputs:
            file.write(" ".join(str(token) for token in input_ids.tolist()))
            file.write("\n")

    # Create the dataset
    dataset = TextDataset(tokenizer=tokenizer, file_path=tokenized_file, block_size=128)

    # Configure the training arguments and other components
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,  # Adjust as needed
        weight_decay=0.01,  # Adjust as needed
        adam_epsilon=1e-8,  # Adjust as needed
        warmup_steps=1000,  # Adjust as needed
        gradient_accumulation_steps=8,  # Adjust as needed
        fp16=True,  # Enable mixed precision training
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(output_dir)