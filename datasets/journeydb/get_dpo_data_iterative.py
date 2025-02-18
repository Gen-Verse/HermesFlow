import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm

bert_model_path = 'google-bert/bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model = BertModel.from_pretrained(bert_model_path)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs['last_hidden_state'][:,0,:].numpy()

def cosine_similarity_bert(s1, s2):
    emb1 = get_bert_embedding(s1)
    emb2 = get_bert_embedding(s2)
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    similarity = 1 - cosine(emb1, emb2)
    return similarity

with open('datasets/journeydb/understanding_caption_results_iter2.json', "r") as f:
    understanding_caption_data_iter2 = json.load(f)

with open('datasets/journeydb/generation_vqa_results_iter2.json', 'r') as f:
    generation_vqa_data_iter2 = json.load(f)

with open('datasets/journeydb/pair_dpo_data.json', "r") as f:
    initial_dpo_data = json.load(f)    
    

for i in tqdm(range(len(understanding_caption_data_iter2))):

    # Understanding preference data
    gt_caption = understanding_caption_data_iter2[i]["caption"]
    bert_score = []
    for j in range(len(understanding_caption_data_iter2[i]["caption_mmu"])):
        bert_score.append(cosine_similarity_bert(gt_caption, understanding_caption_data_iter2[i]["caption_mmu"][j]))
    max_bert_score = max(bert_score)
    max_bert_score_index = bert_score.index(max_bert_score)
    if max_bert_score > initial_dpo_data[i]["bert_score_win"]:
        caption_win = understanding_caption_data_iter2[i]["caption_mmu"][max_bert_score_index]
        caption_lose = initial_dpo_data[i]["caption_win"]
        bert_score_win = max_bert_score
        bert_score_lose = initial_dpo_data[i]["bert_score_win"]
    else:
        caption_win = understanding_caption_data_iter2[i]["caption_mmu"][max_bert_score_index]
        caption_lose = initial_dpo_data[i]["caption_lose"]
        bert_score_win = max_bert_score
        bert_score_lose = initial_dpo_data[i]["bert_score_lose"]

    # Generation preference data
    result = generation_vqa_data_iter2[i]["result"]
    max_result = max(result)
    min_result = min(result)
    if max_result > initial_dpo_data[i]["vqa_score_win"]:
        image_win_path = "datasets/journeydb/generated_images_iter2/" + str(generation_vqa_data_iter2[i]["id"]) + "/" + str(result.index(max_result)) + ".png"
        image_lose_path = initial_dpo_data[i]["image_win"]
        vqa_score_win = max_result
        vqa_score_lose = initial_dpo_data[i]["vqa_score_win"]
    else:
        image_win_path = "datasets/journeydb/generated_images_iter2/" + str(generation_vqa_data_iter2[i]["id"]) + "/" + str(result.index(max_result)) + ".png"
        image_lose_path = initial_dpo_data[i]["image_lose"]
        vqa_score_win = max_result
        vqa_score_lose = initial_dpo_data[i]["vqa_score_lose"]

    with open(f"datasets/journeydb/pair_dpo_data_iter2.json", "a") as f:
        json.dump({
            "id": understanding_caption_data_iter2[i]["id"],
            "img_path": understanding_caption_data_iter2[i]["img_path"],
            "prompt": understanding_caption_data_iter2[i]["prompt"],
            "caption": understanding_caption_data_iter2[i]["caption"],
            "caption_win": caption_win,
            "caption_lose": caption_lose,
            "bert_score_win": bert_score_win,
            "bert_score_lose": bert_score_lose,
            "image_win": image_win_path,
            "image_lose": image_lose_path,
            "vqa_score_win": vqa_score_win,
            "vqa_score_lose": vqa_score_lose
        }, f, ensure_ascii=False, indent=4)
        f.write(',\n')