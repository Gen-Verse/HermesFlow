import copy
import json
import os
from functools import partial

import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
# from training.utils import image_transform
from llava.llava import conversation as conversation_lib

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."


from torchvision import transforms
def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

def preprocess_multimodal(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # Customized operation, get rid of <image> special token. Edited by Zechen
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")
                sentence['value'] = sentence['value'].strip()

    return sources


def preprocess_v0(
        sources,
        tokenizer,
):
    # Let's assume has_image is false, since we will process the image token separately
    has_image = False

    # Adapted from llava-phi/mipha/train/train.py
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = 0                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    input_ids_system = tokenizer(
        [SYSTEM_PROMPT for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_system=input_ids_system
    )


class LLaVADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 phase,
                 ):
        super(LLaVADataset, self).__init__()

        self.tokenizer = tokenizer

        if phase == "pretrain":
            data_file_path = "/mnt/bn/vgfm2/test_dit/blip_laion_cc_sbu_558k.json"
            self.image_root = "/mnt/bn/vgfm2/test_dit/pretraining_data"
        else:
            data_file_path = "datasets/understanding_data_try4.json"
            self.image_root = "/mnt/bn/vgfm2/test_dit/tuning_data"

        with open(data_file_path, 'r') as f:
            data = json.load(f)
        self.list_data_dict = []
        for item in data:
            if 'image1' in item.keys():
                self.list_data_dict.append(item)

        print("Formatting llava instruction data")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        assert 'image1' in sources[0]
        image1_path = self.list_data_dict[i]['image1']
        image2_path = self.list_data_dict[i]['image2']
        try:
            image1 = Image.open(image1_path).convert('RGB')
            image1 = image_transform(image1)
            image2 = Image.open(image2_path).convert('RGB')
            image2 = image_transform(image2)
        except:
            print("Read image error. Use dummy data.")
            crop_size = 256
            image1 = image2 = torch.zeros(3, crop_size, crop_size)

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))

        data_dict = preprocess_v0(sources, self.tokenizer)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0])

        # image exist in the data
        if 'image1' in self.list_data_dict[i] and 'image2' in self.list_data_dict[i]:
            data_dict['image1'] = image1
            data_dict['image2'] = image2
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = 256
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        return data_dict


def collate_fn(
        instances,
        tokenizer=None,
        max_length=77,
):
    input_ids, labels, input_ids_system = tuple([instance[key] for instance in instances]
                                                for key in ("input_ids", "labels", "input_ids_system"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id) # torch.Size([1, 455])
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX) # torch.Size([1, 455])
    input_ids_system = torch.stack(input_ids_system, dim=0) # torch.Size([1, 28])

    offset = max_length - input_ids.shape[-1] - input_ids_system.shape[-1]

    if input_ids.shape[-1] < max_length - input_ids_system.shape[-1]:
        pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
        input_ids = torch.cat([input_ids, pad_tube], dim=1)

        pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
        labels = torch.cat([labels, pad_tube], dim=1)

    min_max_len = min(
        max_length - input_ids_system.shape[-1],
        tokenizer.model_max_length - input_ids_system.shape[-1],
    )

    input_ids = input_ids[:, :min_max_len]
    labels = labels[:, :min_max_len]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        input_ids_system=input_ids_system,
    )

    if 'image1' in instances[0] and 'image2' in instances[0]:
        images1 = [instance['image1'] for instance in instances]
        images2 = [instance['image2'] for instance in instances]
        if all(x is not None and x.shape == images1[0].shape for x in images1):
            batch['images1'] = torch.stack(images1)
        else:
            batch['images1'] = images1
        if all(x is not None and x.shape == images2[0].shape for x in images2):
            batch['images2'] = torch.stack(images2)
        else:
            batch['images2'] = images2

    return batch


def get_instruct_data_loader(
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
        phase,
):
    train_dataset = LLaVADataset(
        tokenizer,
        phase,
    )
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader

class DPODataset(Dataset):

    def __init__(self,
                 tokenizer,
                 phase,
                 ):
        super(DPODataset, self).__init__()

        self.tokenizer = tokenizer
        data_file_path = "datasets/understanding_dpo_data.json"
        with open(data_file_path, 'r') as f:
            data = json.load(f)
        self.list_data_dict = []
        for item in data:
            self.list_data_dict.append(item)

        print("Formatting dpo data")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        image_initial_path = self.list_data_dict[i]['image']['initial']
        image_optimized_path = self.list_data_dict[i]['image']['optimized']

        image_initial = Image.open(image_initial_path).convert('RGB')
        image_initial = image_transform(image_initial)
        image_optimized = Image.open(image_optimized_path).convert('RGB')
        image_optimized = image_transform(image_optimized)

        conversation_lose_initial = [
            {
                "from": "human",
                "value": "Give a detailed caption for this image."
            },
            {
                "from": "gpt",
                "value": sources[0]["caption_lose"]["caption_initial"]
            }
        ]
        conversation_lose_optimized = [
            {
                "from": "human",
                "value": "Give a detailed caption for this image."
            },
            {
                "from": "gpt",
                "value": sources[0]["caption_lose"]["caption_optimized"]
            }
        ]
        conversation_win_initial = [
            {
                "from": "human",
                "value": "Give a detailed caption for this image."
            },
            {
                "from": "gpt",
                "value": sources[0]["caption_win"]["caption_initial"]
            }
        ]
        conversation_win_optimized = [
            {
                "from": "human",
                "value": "Give a detailed caption for this image."
            },
            {
                "from": "gpt",
                "value": sources[0]["caption_win"]["caption_optimized"]
            }
        ]
        sources_lose_initial = preprocess_multimodal([conversation_lose_initial])
        sources_lose_optimized = preprocess_multimodal([conversation_lose_optimized])
        sources_win_initial = preprocess_multimodal([conversation_win_initial])
        sources_win_optimized = preprocess_multimodal([conversation_win_optimized])
        data_dict_lose_initial = preprocess_v0(sources_lose_initial, self.tokenizer)
        data_dict_lose_optimized = preprocess_v0(sources_lose_optimized, self.tokenizer)
        data_dict_win_initial = preprocess_v0(sources_win_initial, self.tokenizer)
        data_dict_win_optimized = preprocess_v0(sources_win_optimized, self.tokenizer)

        data_dict_lose_initial = dict(input_ids=data_dict_lose_initial["input_ids"][0],
                            labels=data_dict_lose_initial["labels"][0],
                            input_ids_system=data_dict_lose_initial["input_ids_system"][0])
        data_dict_lose_optimized = dict(input_ids=data_dict_lose_optimized["input_ids"][0],
                            labels=data_dict_lose_optimized["labels"][0],
                            input_ids_system=data_dict_lose_optimized["input_ids_system"][0])
        data_dict_win_initial = dict(input_ids=data_dict_win_initial["input_ids"][0],
                            labels=data_dict_win_initial["labels"][0],
                            input_ids_system=data_dict_win_initial["input_ids_system"][0])
        data_dict_win_optimized = dict(input_ids=data_dict_win_optimized["input_ids"][0],
                            labels=data_dict_win_optimized["labels"][0],
                            input_ids_system=data_dict_win_optimized["input_ids_system"][0])
        max_seq_len_lose_input_ids = max(data_dict_lose_initial["input_ids"].size(0), data_dict_lose_optimized["input_ids"].size(0))
        data_dict_lose_initial["input_ids"] = torch.nn.functional.pad(data_dict_lose_initial["input_ids"], (0, max_seq_len_lose_input_ids - data_dict_lose_initial["input_ids"].size(0)), value=self.tokenizer.pad_token_id)
        data_dict_lose_optimized["input_ids"] = torch.nn.functional.pad(data_dict_lose_optimized["input_ids"], (0, max_seq_len_lose_input_ids - data_dict_lose_optimized["input_ids"].size(0)), value=self.tokenizer.pad_token_id)

        max_seq_len_win_input_ids = max(data_dict_win_initial["input_ids"].size(0), data_dict_win_optimized["input_ids"].size(0))
        data_dict_win_initial["input_ids"] = torch.nn.functional.pad(data_dict_win_initial["input_ids"], (0, max_seq_len_win_input_ids - data_dict_win_initial["input_ids"].size(0)), value=self.tokenizer.pad_token_id)
        data_dict_win_optimized["input_ids"] = torch.nn.functional.pad(data_dict_win_optimized["input_ids"], (0, max_seq_len_win_input_ids - data_dict_win_optimized["input_ids"].size(0)), value=self.tokenizer.pad_token_id)

        max_seq_len_lose_labels = max(data_dict_lose_initial["labels"].size(0), data_dict_lose_optimized["labels"].size(0))
        data_dict_lose_initial["labels"] = torch.nn.functional.pad(data_dict_lose_initial["labels"], (0, max_seq_len_lose_labels - data_dict_lose_initial["labels"].size(0)), value=IGNORE_INDEX)
        data_dict_lose_optimized["labels"] = torch.nn.functional.pad(data_dict_lose_optimized["labels"], (0, max_seq_len_lose_labels - data_dict_lose_optimized["labels"].size(0)), value=IGNORE_INDEX)

        max_seq_len_win_labels = max(data_dict_win_initial["labels"].size(0), data_dict_win_optimized["labels"].size(0))
        data_dict_win_initial["labels"] = torch.nn.functional.pad(data_dict_win_initial["labels"], (0, max_seq_len_win_labels - data_dict_win_initial["labels"].size(0)), value=IGNORE_INDEX)
        data_dict_win_optimized["labels"] = torch.nn.functional.pad(data_dict_win_optimized["labels"], (0, max_seq_len_win_labels - data_dict_win_optimized["labels"].size(0)), value=IGNORE_INDEX)

        lose_dict = {
            key: torch.stack([data_dict_lose_initial[key], data_dict_lose_optimized[key]], dim=0)
            for key in data_dict_lose_initial.keys()
        }
        win_dict = {
            key: torch.stack([data_dict_win_initial[key], data_dict_win_optimized[key]], dim=0)
            for key in data_dict_win_initial.keys()
        }

        image = torch.stack([image_initial, image_optimized], dim=0)
        data_dict = {
            "image": image,
            "lose_dict": lose_dict,
            "win_dict": win_dict
        }

        return data_dict


def collate_fn_dpo(
        instances,
        tokenizer=None,
        max_length=77,
):

    win_input_ids, win_labels, win_input_ids_system = tuple(
        [instance["win_dict"][key] for instance in instances] for key in ("input_ids", "labels", "input_ids_system")
    )

    lose_input_ids, lose_labels, lose_input_ids_system = tuple(
        [instance["lose_dict"][key] for instance in instances] for key in ("input_ids", "labels", "input_ids_system")
    )

    max_seq_len_win = max(x.size(1) for x in win_input_ids)
    padded_win_input_ids = [
        torch.nn.functional.pad(x, (0, max_seq_len_win - x.size(1)), value=tokenizer.pad_token_id)
        for x in win_input_ids
    ]
    win_input_ids = torch.stack(padded_win_input_ids, dim=0)

    max_seq_len_win_labels = max(x.size(1) for x in win_labels)
    padded_win_labels = [
        torch.nn.functional.pad(x, (0, max_seq_len_win_labels - x.size(1)), value=IGNORE_INDEX)
        for x in win_labels
    ]
    win_labels = torch.stack(padded_win_labels, dim=0)

    win_input_ids_system = torch.stack(win_input_ids_system, dim=0)


    max_seq_len_lose = max(x.size(1) for x in lose_input_ids)
    padded_lose_input_ids = [
        torch.nn.functional.pad(x, (0, max_seq_len_lose - x.size(1)), value=tokenizer.pad_token_id)
        for x in lose_input_ids
    ]
    lose_input_ids = torch.stack(padded_lose_input_ids, dim=0)

    max_seq_len_lose_labels = max(x.size(1) for x in lose_labels)
    padded_lose_labels = [
        torch.nn.functional.pad(x, (0, max_seq_len_lose_labels - x.size(1)), value=IGNORE_INDEX)
        for x in lose_labels
    ]
    lose_labels = torch.stack(padded_lose_labels, dim=0)

    lose_input_ids_system = torch.stack(lose_input_ids_system, dim=0)

    # 分别计算 win 和 lose 的填充长度
    offset_win = max_length - win_input_ids.shape[-1] - win_input_ids_system.shape[-1]
    offset_lose = max_length - lose_input_ids.shape[-1] - lose_input_ids_system.shape[-1]

    # 针对 win 部分的填充
    if win_input_ids.shape[-1] < max_length - win_input_ids_system.shape[-1]:
        pad_tube = torch.ones(size=(win_input_ids.shape[0], 2, offset_win), dtype=win_input_ids.dtype) * tokenizer.pad_token_id
        win_input_ids = torch.cat([win_input_ids, pad_tube], dim=2)

        pad_tube = torch.ones(size=(win_labels.shape[0], 2, offset_win), dtype=win_labels.dtype) * IGNORE_INDEX
        win_labels = torch.cat([win_labels, pad_tube], dim=2)

    # 针对 lose 部分的填充
    if lose_input_ids.shape[-1] < max_length - lose_input_ids_system.shape[-1]:
        pad_tube = torch.ones(size=(lose_input_ids.shape[0], 2, offset_lose), dtype=lose_input_ids.dtype) * tokenizer.pad_token_id
        lose_input_ids = torch.cat([lose_input_ids, pad_tube], dim=2)

        pad_tube = torch.ones(size=(lose_labels.shape[0], 2, offset_lose), dtype=lose_labels.dtype) * IGNORE_INDEX
        lose_labels = torch.cat([lose_labels, pad_tube], dim=2)

    # 分别计算 win 和 lose 的截断长度
    min_max_len_win = min(
        max_length - win_input_ids_system.shape[-1],
        tokenizer.model_max_length - win_input_ids_system.shape[-1],
    )

    min_max_len_lose = min(
        max_length - lose_input_ids_system.shape[-1],
        tokenizer.model_max_length - lose_input_ids_system.shape[-1],
    )

    # 针对 win 和 lose 的截断
    win_input_ids = win_input_ids[:, :, :min_max_len_win]
    win_labels = win_labels[:, :, :min_max_len_win]

    lose_input_ids = lose_input_ids[:, :, :min_max_len_lose]
    lose_labels = lose_labels[:, :, :min_max_len_lose]

    # 将 win 和 lose 数据合并到 batch 中
    batch = dict(
        win_dict=dict(
            input_ids=win_input_ids,
            labels=win_labels,
            attention_mask=win_input_ids.ne(tokenizer.pad_token_id),
            input_ids_system=win_input_ids_system,
        ),
        lose_dict=dict(
            input_ids=lose_input_ids,
            labels=lose_labels,
            attention_mask=lose_input_ids.ne(tokenizer.pad_token_id),
            input_ids_system=lose_input_ids_system,
        )
    )

    images = [instance['image'] for instance in instances]
    batch['images'] = torch.stack(images)
    return batch

def get_dpo_data_loader(
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
        phase,
):
    train_dataset = DPODataset(
        tokenizer,
        phase,
    )
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn_dpo,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader


############################################################################################################
class UniDPODatasetUnderstanding(Dataset):

    def __init__(self,
                 tokenizer,
                 phase,
                 ):
        super(UniDPODatasetUnderstanding, self).__init__()

        self.tokenizer = tokenizer

        data_file_path = "datasets/journeydb/dpo_data.json"

        with open(data_file_path, 'r') as f:
            data = json.load(f)
        self.list_data_dict = []
        for item in data:
            self.list_data_dict.append(item)

        print("Formatting llava instruction data")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        image = Image.open(self.list_data_dict[i]['img_path']).convert('RGB')
        image = image_transform(image)

        conversation_lose = [
            {
                "from": "human",
                "value": "Give a detailed caption for this image."
            },
            {
                "from": "gpt",
                "value": sources[0]["caption_lose"]
            }
        ]
        conversation_win = [
            {
                "from": "human",
                "value": "Give a detailed caption for this image."
            },
            {
                "from": "gpt",
                "value": sources[0]["caption_win"]
            }
        ]
        sources_lose = preprocess_multimodal([conversation_lose])
        sources_win = preprocess_multimodal([conversation_win])

        data_dict_lose = preprocess_v0(sources_lose, self.tokenizer)
        data_dict_win = preprocess_v0(sources_win, self.tokenizer)

        if isinstance(i, int):
            data_dict = dict(input_ids_lose=data_dict_lose["input_ids"][0],
                             labels_lose=data_dict_lose["labels"][0],
                             input_ids_system_lose=data_dict_lose["input_ids_system"][0],
                             input_ids_win=data_dict_win["input_ids"][0],
                             labels_win=data_dict_win["labels"][0],
                             input_ids_system_win=data_dict_win["input_ids_system"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = 256
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        return data_dict


def collate_fn_understanding(
        instances,
        tokenizer=None,
        max_length=77,
):
    input_ids_lose, labels_lose, input_ids_system_lose, input_ids_win, labels_win, input_ids_system_win = tuple(
        [instance[key] for instance in instances]
        for key in ("input_ids_lose", "labels_lose", "input_ids_system_lose", "input_ids_win", "labels_win", "input_ids_system_win")
    )

    # Padding for lose-related inputs
    input_ids_lose = torch.nn.utils.rnn.pad_sequence(
        input_ids_lose,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels_lose = torch.nn.utils.rnn.pad_sequence(labels_lose,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
    input_ids_system_lose = torch.stack(input_ids_system_lose, dim=0)

    # Padding for win-related inputs
    input_ids_win = torch.nn.utils.rnn.pad_sequence(
        input_ids_win,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels_win = torch.nn.utils.rnn.pad_sequence(labels_win,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
    input_ids_system_win = torch.stack(input_ids_system_win, dim=0)

    # Determine offset for lose-related inputs
    offset_lose = max_length - input_ids_lose.shape[-1] - input_ids_system_lose.shape[-1]

    if input_ids_lose.shape[-1] < max_length - input_ids_system_lose.shape[-1]:
        pad_tube_lose = torch.ones(size=(input_ids_lose.shape[0], offset_lose), dtype=input_ids_lose.dtype) * tokenizer.pad_token_id
        input_ids_lose = torch.cat([input_ids_lose, pad_tube_lose], dim=1)

        pad_tube_lose = torch.ones(size=(labels_lose.shape[0], offset_lose), dtype=labels_lose.dtype) * IGNORE_INDEX
        labels_lose = torch.cat([labels_lose, pad_tube_lose], dim=1)

    min_max_len_lose = min(
        max_length - input_ids_system_lose.shape[-1],
        tokenizer.model_max_length - input_ids_system_lose.shape[-1],
    )

    input_ids_lose = input_ids_lose[:, :min_max_len_lose]
    labels_lose = labels_lose[:, :min_max_len_lose]

    # Determine offset for win-related inputs
    offset_win = max_length - input_ids_win.shape[-1] - input_ids_system_win.shape[-1]

    if input_ids_win.shape[-1] < max_length - input_ids_system_win.shape[-1]:
        pad_tube_win = torch.ones(size=(input_ids_win.shape[0], offset_win), dtype=input_ids_win.dtype) * tokenizer.pad_token_id
        input_ids_win = torch.cat([input_ids_win, pad_tube_win], dim=1)

        pad_tube_win = torch.ones(size=(labels_win.shape[0], offset_win), dtype=labels_win.dtype) * IGNORE_INDEX
        labels_win = torch.cat([labels_win, pad_tube_win], dim=1)

    min_max_len_win = min(
        max_length - input_ids_system_win.shape[-1],
        tokenizer.model_max_length - input_ids_system_win.shape[-1],
    )

    input_ids_win = input_ids_win[:, :min_max_len_win]
    labels_win = labels_win[:, :min_max_len_win]

    batch = dict(
        input_ids_lose=input_ids_lose,
        labels_lose=labels_lose,
        attention_mask_lose=input_ids_lose.ne(tokenizer.pad_token_id),
        input_ids_system_lose=input_ids_system_lose,
        input_ids_win=input_ids_win,
        labels_win=labels_win,
        attention_mask_win=input_ids_win.ne(tokenizer.pad_token_id),
        input_ids_system_win=input_ids_system_win,
    )

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images

    return batch


def get_unidpo_understanding_data_loader(
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
        phase,
):
    train_dataset = UniDPODatasetUnderstanding(
        tokenizer,
        phase,
    )
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn_understanding,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader

if __name__ == '__main__':
    import transformers
    pretrained_model_path = '/mnt/bn/vgfm2/test_mlx/xavier/pretrained_weights/phi-1_5'
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_path,
                                                           padding_side="left")
    special_tokens = ("soi", "eoi", "sovi", "eovi", "t2i", "mmu", "t2v", "v2v", "lvg")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(list(special_tokens))

    dataset = LLaVADataset(
        tokenizer,
        "tuning"
    )

    item = dataset.__getitem__(0)
    import pdb
    pdb.set_trace()

