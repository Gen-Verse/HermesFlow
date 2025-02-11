import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import Showo, MAGVITv2, CLIPVisionTower
from training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu, create_attention_mask_for_mmu_vit
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
from transformers import CLIPImageProcessor
import json

from llava.llava import conversation as conversation_lib

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_PROMPT_LEN = 28

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name + '_mmu',
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    vision_tower_name = "openai/clip-vit-large-patch14-336"
    vision_tower =  CLIPVisionTower(vision_tower_name).to(device)
    clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model.eval()

    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability

    with open('datasets/journeydb/initial_data.json', "r") as f:
        data = json.load(f)

    results = []
    caption = []
    for i in tqdm(range(len(data))):
        image_path = data[i]["img_path"]
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)

        pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]

        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1
        question = "Give a caption for this image."
        
        input_ids = uni_prompting.text_tokenizer(['USER: \n' + question + ' ASSISTANT:'])[
            'input_ids']
        input_ids = torch.tensor(input_ids).to(device)

        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
            input_ids
        ], dim=1).long()

        attention_mask = create_attention_mask_for_mmu(input_ids.to(device),
                                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
        for top_k in [1, 2, 3, 4]:
            if top_k != 1:
                for tempterature in [0.1, 0.7, 2.0]:
                    cont_toks_list = model.mmu_generate(input_ids, attention_mask=attention_mask,
                                                        max_new_tokens=config.max_new_tokens, top_k=top_k,
                                                        temperature=tempterature,
                                                        eot_token=uni_prompting.sptids_dict['<|eot|>'])

                    cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

                    text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
                    caption.append(text[0])
            else:
                tempterature = 1
                cont_toks_list = model.mmu_generate(input_ids, attention_mask=attention_mask,
                                                        max_new_tokens=config.max_new_tokens, top_k=top_k,
                                                        temperature=tempterature,
                                                        eot_token=uni_prompting.sptids_dict['<|eot|>'])

                cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

                text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
                caption.append(text[0])
        with open(f"datasets/journeydb/understanding_caption_results", "a") as f:
            json.dump({
                "id": data[i]["id"],
                "img_path": data[i]["img_path"],
                "prompt": data[i]["prompt"],
                "caption": data[i]["caption"],
                "caption_mmu": caption
            }, f, ensure_ascii=False, indent=4)
            f.write(',\n')
        caption = []



