from tifascore import get_llama2_pipeline, get_llama2_question_and_answers, UnifiedQAModel, filter_question_and_answers
import json
from tqdm import tqdm

pipeline = get_llama2_pipeline("tifa-benchmark/llama2_tifa_question_generation")
unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")

with open("datasets/journeydb/initial_data.json", "r") as f:
    data = json.load(f) 

for i in tqdm(range(len(data))):
    caption = data[i]['caption']
    llama2_questions = get_llama2_question_and_answers(pipeline, caption)
    filtered_questions = filter_question_and_answers(unifiedqa_model, llama2_questions)
    for dict_ in filtered_questions:
        dict_['id'] = data[i]['id']
        dict_['img_path'] = data[i]['img_path']
        dict_['prompt'] = data[i]["prompt"]

        with open(f"datasets/journeydb/vqa_data.json", "a") as f:
            json.dump(dict_, f, ensure_ascii=False, indent=4)
            f.write(',\n')
