from transformers import Trainer
import torch.nn.functional as F


# https://discuss.huggingface.co/t/multi-label-token-classification/16509/7
class MultiLabelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Use BCEWithLogitsLoss for multi-label classification
        logits = logits[labels!=-100]
        labels = labels[labels!=-100]
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels.float()
        )

        return (loss, outputs) if return_outputs else loss
