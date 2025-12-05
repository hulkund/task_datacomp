from peft import get_peft_model, LoraConfig, PeftModel
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
import clip
import numpy as np
import timm # for running vit

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


def get_model_processor(finetune_type):
    if finetune_type=="linear_probe":
        model, preprocess = clip.load('ViT-B/32', "cuda")
        return model, preprocess
    elif finetune_type=="full_finetune_resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        preprocess = weights.transforms()
        return model, preprocess
    elif finetune_type=="full_finetune_resnet101":
        weights = ResNet101_Weights.DEFAULT
        model = resnet101(weights=weights)
        preprocess = weights.transforms()
        return model, preprocess
    elif finetune_type=="lora_finetune_vit":
        timm_model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        model = get_lora_model(timm_model)
        data_config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_config, is_training=True, no_aug=True)
        return model, preprocess        

def get_features(dataset_name, split, subset_path):
    device="cuda"
    if 'task' in split:
        split='test'+split[4]
    embed=np.load(f'/data/vision/beery/scratch/neha/task-datacomp/all_datasets/{dataset_name}//embeddings/{split}_image_label_embed.npz')
    if subset_path:
        subset=np.load(subset_path,allow_pickle=True)
        id_to_index = {id_: idx for idx, id_ in enumerate(subset)}
        mask = [id_to_index[id_] for id_ in subset if id_ in id_to_index]
        features=embed['features'][mask]
        labels=embed['labels'][mask]
        print("training on x samples", len(features))
    else:
        features=embed['features']
        labels=embed['labels']
    return features, labels
