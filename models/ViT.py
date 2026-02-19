from peft import get_peft_model, LoraConfig, PeftModel
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
import clip
import numpy as np

class ViTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ViTModel(PreTrainedModel):
    config_class = ViTConfig

    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        self.blocks = model.blocks

    def forward(self, x):
        return self.model(x)

def get_lora_model(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.01,
        bias='none',
        target_modules=['qkv'],
        modules_to_save=["classifier"],
    )
    extractor_model = get_peft_model(ViTModel(model, ViTConfig()), config).to('cuda')
    return extractor_model