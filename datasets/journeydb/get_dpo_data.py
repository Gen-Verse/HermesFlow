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

with open('datasets/journeydb/understanding_caption_results.json', "r") as f:
    understanding_caption_data = json.load(f)

with open('datasets/journeydb/generation_vqa_results.json', 'r') as f:
    generation_vqa_data = json.load(f)
    
for i in tqdm(range(len(understanding_caption_data))):

    # Understanding preference data
    gt_caption = understanding_caption_data[i]["caption"]
    bert_score = []
    for j in range(len(understanding_caption_data[i]["caption_mmu"])):
        bert_score.append(cosine_similarity_bert(gt_caption, understanding_caption_data[i]["caption_mmu"][j]))
    max_bert_score = max(bert_score)
    max_bert_score_index = bert_score.index(max_bert_score)
    min_bert_score = min(bert_score)
    min_bert_score_index = bert_score.index(min_bert_score)

    # Generation preference data
    result = generation_vqa_data[i]["result"]
    max_result = max(result)
    min_result = min(result)
    if max_result != min_result and max_result > 0.6:
        max_index = result.index(max_result)
        min_index = result.index(min_result)
    with open(f"datasets/journeydb/pair_dpo_data.json", "a") as f:
        json.dump({
            "id": understanding_caption_data[i]["id"],
            "img_path": understanding_caption_data[i]["img_path"],
            "prompt": understanding_caption_data[i]["prompt"],
            "caption": understanding_caption_data[i]["caption"],
            "caption_win": understanding_caption_data[i]["caption_mmu"][max_bert_score_index],
            "caption_lose": understanding_caption_data[i]["caption_mmu"][min_bert_score_index],
            "bert_score_win": max_bert_score,
            "bert_score_lose": min_bert_score,
            "image_win": "datasets/journeydb/generated_images/"+str(understanding_caption_data[i]["id"])+"/"+str(max_index)+".png",
            "image_lose": "datasets/journeydb/generated_images/"+str(understanding_caption_data[i]["id"])+"/"+str(min_index)+".png",
            "vqa_score_win": max_result,
            "vqa_score_lose": min_result
        }, f, ensure_ascii=False, indent=4)
        f.write(',\n')




