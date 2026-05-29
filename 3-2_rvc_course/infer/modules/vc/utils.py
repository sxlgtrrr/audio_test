import os

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

USE_FAIRSEQ = False
try:
    from fairseq import checkpoint_utils

    USE_FAIRSEQ = True
except Exception:
    from transformers import HubertModel

HUBERT_MODEL_ID = "facebook/hubert-base-ls960"
HUBERT_LOCAL_DIR = os.path.join("assets", "hubert", "hubert-base-ls960")


def _load_transformers_hubert():
    if os.path.isdir(HUBERT_LOCAL_DIR):
        try:
            return HubertModel.from_pretrained(HUBERT_LOCAL_DIR, local_files_only=True)
        except Exception:
            pass
    try:
        model = HubertModel.from_pretrained(HUBERT_MODEL_ID, local_files_only=True)
    except Exception:
        model = HubertModel.from_pretrained(HUBERT_MODEL_ID)
    os.makedirs(HUBERT_LOCAL_DIR, exist_ok=True)
    model.save_pretrained(HUBERT_LOCAL_DIR)
    return model


class HubertInferenceModel:
    def __init__(self, model, version, device, is_half):
        self.model = model
        self.version = version
        self.device = device
        self.is_half = is_half
        self.final_proj = getattr(model, "final_proj", None)

    def extract_features(self, source, padding_mask, output_layer):
        if USE_FAIRSEQ:
            return self.model.extract_features(
                source=source,
                padding_mask=padding_mask,
                output_layer=output_layer,
            )
        with torch.no_grad():
            outputs = self.model(source, output_hidden_states=True)
            return outputs.hidden_states[output_layer], None

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

    def half(self):
        self.model = self.model.half()
        self.is_half = True
        return self

    def float(self):
        self.model = self.model.float()
        self.is_half = False
        return self

    def eval(self):
        self.model.eval()
        return self


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    if USE_FAIRSEQ:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["assets/hubert/hubert_base.pt"],
            suffix="",
        )
        hubert_model = models[0]
    else:
        hubert_model = _load_transformers_hubert()

    wrapper = HubertInferenceModel(
        hubert_model,
        version=getattr(config, "version", "v2"),
        device=config.device,
        is_half=config.is_half,
    )
    wrapper = wrapper.to(config.device)
    if config.is_half:
        wrapper = wrapper.half()
    else:
        wrapper = wrapper.float()
    return wrapper.eval()
