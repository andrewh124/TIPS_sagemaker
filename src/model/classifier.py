# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from transformers import AutoModel, PreTrainedModel

# class IdeaDetectionClassifier(PreTrainedModel):
#     def __init__(self, model_name: str, n_labels: int):
#         super().__init__(config=None)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
#         self.classifier = nn.Linear(self.model.config.hidden_size, n_labels)

#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[1]  # Get the pooled output (CLS token representation)
#         logits = self.classifier(pooled_output)
#         return logits
