import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def set_seed(seed):
  """Sets the seed for reproducibility."""
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set seed for reproducibility
  set_seed(42)

  # Load model, tokenizer, and select device
  model, tokenizer = load_paraphrasing_model()
  device = get_device()
  model.to(device)

def load_paraphrasing_model(model_name='ramsrigouthamg/t5_paraphraser'):
  """Loads the T5 paraphrasing model and tokenizer."""
  model = T5ForConditionalGeneration.from_pretrained(model_name)
  tokenizer = T5Tokenizer.from_pretrained(model_name)
  return model, tokenizer

def get_device():
  """Determines and returns the best available device (GPU or CPU)."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  return device

def paraphrase_sentence(sentence):
    paraphrase_sentence(model, tokenizer, device, sentence)

def paraphrase_sentence(model, tokenizer, device, sentence, num_return_sequences=10, max_len=256):
  """
  Generates paraphrased versions of a given sentence.

  Args:
    model: The pre-trained T5 model.
    tokenizer: The T5 tokenizer.
    device: The device to run the model on (cuda or cpu).
    sentence: The sentence to paraphrase.
    num_return_sequences: The number of paraphrased sentences to return.
    max_len: The maximum length of the generated sequences.

  Returns:
    A list of unique, paraphrased sentences.
  """
  text = f"paraphrase: {sentence} </s>"
  encoding = tokenizer.encode_plus(
      text,
      max_length=max_len,
      pad_to_max_length=True,
      return_tensors="pt"
  )
  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

  beam_outputs = model.generate(
      input_ids=input_ids,
      attention_mask=attention_masks,
      do_sample=True,
      max_length=max_len,
      top_k=120,
      top_p=0.98,
      early_stopping=True,
      num_return_sequences=num_return_sequences
  )

  final_outputs = []
  for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if sent.lower() != sentence.lower() and sent not in final_outputs:
      final_outputs.append(sent)
  
  return final_outputs

def main():
  """The main function to orchestrate the paraphrasing process."""
  # Set seed for reproducibility
  set_seed(42)

  # Load model, tokenizer, and select device
  model, tokenizer = load_paraphrasing_model()
  device = get_device()
  model.to(device)

  # Define the sentence to paraphrase
  sentence = "Which course should I take to get started in data science?"
  #sentence  = "As AI shapes human decisions, does algorithmic morality reduce responsibility? Examine culture, bias, emotion, and empathy in balancing tech efficiency."
  # Generate paraphrases
  paraphrased_questions = paraphrase_sentence(model, tokenizer, device, sentence)

  # Print results
  print("\nOriginal Question ::")
  print(sentence)
  print("\nParaphrased Questions :: ")
  for i, final_output in enumerate(paraphrased_questions):
      print(f"{i+1}: {final_output}")

if __name__ == "__main__":
  main()
