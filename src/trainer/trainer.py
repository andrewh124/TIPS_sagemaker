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
        attention_mask = (
            inputs["attention_mask"].unsqueeze(-1).repeat(1, 1, logits.size(-1))
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), weight=attention_mask, reduction="none"
        )
        loss = loss.sum() / (inputs["attention_mask"].sum() + 0.0000001)

        return (loss, outputs) if return_outputs else loss
