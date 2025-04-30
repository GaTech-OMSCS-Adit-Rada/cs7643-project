import torch.nn as nn
class NLEWithGPT(nn.Module):
    def __init__(self, gpt_model, num_labels=3):
        super().__init__()
        self.gpt_model = gpt_model
        self.gpt_model.out_head = nn.Linear(gpt_model.hidden_size, num_labels)

    def forward(self, input_ids):
        logits = self.gpt_model(input_ids)
        # out = logits.mean(dim=1)
        return logits