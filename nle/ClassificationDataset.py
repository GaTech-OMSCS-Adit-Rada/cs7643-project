
import torch
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=None, pad_token_id=50256):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the dataset with columns like 'sentence1', 'sentence2', and 'gold_label'.
            tokenizer: Tokenizer to encode the input text.
        """
        self.data = data
        self.tokenizer = tokenizer

        # Preprocess the data to store tokenized inputs and labels
        self.encoded_texts = []
        self.labels = []

        # Map the label to an integer (e.g., 0 for entailment, 1 for contradiction, 2 for neutral)
        label_mapping = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        
        for index, entry in data.iterrows():
            # Format the input text
            instruction_plus_input = format_input(entry)
            # Tokenize the input text
            tokenized_input = tokenizer.encode(instruction_plus_input)

            # Get the label (e.g., 'entailment', 'contradiction', 'neutral')
            label = entry['gold_label']

            # Store the tokenized input and label
            self.encoded_texts.append(tokenized_input)
            self.labels.append(label_mapping[label])
                
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        # Truncate sequences if they are longer than max_length
        self.encoded_texts = [
            encoded_text[:self.max_length] if len(encoded_text) > 0 else [0]
            for encoded_text in self.encoded_texts
        ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
        

    def __getitem__(self, index):
        """
        Returns:
            dict: A dictionary containing 'input_ids' (tokenized input) and 'label' (classification label).
        """
        # return {
        #     'input_ids': self.encoded_texts[index],
        #     'label': self.labels[index]
        # }
        encoded = self.encoded_texts[index]
        label = self.labels[index]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    
def format_input(entry):
    instruction_text = (
        # f"Below is a natural language entailment data classification. "
        # f"Write a response that appropriately determines the medical relationship between a premise (sentence 1) and a hypothesis (sentence 2)"
        f"\n\n### Premise:\n{entry['sentence1']}"
        f"\n\n### hypothesis:\n{entry['sentence2']}"
    )

    return instruction_text

def format_response(entry):
    return entry['gold_label']

def classification_collate_fn(batch, tokenizer,allowed_max_length, pad_token_id=50256, device="cpu"):
    # for entry in batch:
    #     print(entry)
    input_ids = [entry['input_ids'] for entry in batch]
    labels = [entry['label'] for entry in batch]

    # Find the maximum sequence length in the batch
    max_length = max(len(ids) for ids in input_ids)

    # If allowed_max_length is specified, truncate sequences to this length
    # if allowed_max_length is not None:
    #     max_length = min(max_length, allowed_max_length)

    # Pad input IDs to the maximum length
    input_ids = [
        ids[:max_length] + [pad_token_id] * (allowed_max_length - len(ids[:max_length]))
        for ids in input_ids
    ]

    input_ids = torch.tensor(input_ids).to(device)
    labels = torch.tensor(labels).to(device)

    return input_ids, labels