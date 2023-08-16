import json
import os.path
import time
import openai

# Use the openai library to set the API key
# openai.api_key = "sk-"
gpt_version = 'gpt-3.5-turbo'
openai.api_base = ""

def gpt_35_api_stream(messages: list):
    try:
        response = openai.ChatCompletion.create(
            model=gpt_version,
            messages=messages,
            stream=True,
        )
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                # print(f' {completion}')
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                # print(f'{delta_k} = {delta_v}')
                completion[delta_k] += delta_v
        messages.append(completion)  
        return (True, '')
    except Exception as err:
        return (False, f'OpenAI API error: {err}')

if __name__ == '__main__':
    sys_message = {"role": "system",
                   "content": "You are an expert in the medical field. Now you have some questions and corresponding "
                              "answer candidate for medical images. Please answer the question according "
                              "to the answer candidates. Each answer candidate is associated with a confidence "
                              "score within a bracket. Keywords may not be relevant to the question."
                              "The true answer may not be included in the candidates. \n"
                              "Keywords: cardiovascular(0.05), chest(0.02), myocardium(0.02)\n"
                              "Question: Is there cardiomyopathy?\n"
                              "Candidates: yes(0.990), no(0.002), stress(0.00), there(0.00), 12(0.00)\n"
                              "Answer: yes"
                   }

    root = './results'
    splits = ['pathvqa', 'radvqa', 'slakevqa']
    method = 'biomed'

    for split in splits:
        last_qids = []
        try:
            with open(os.path.join(root, f'(3){split}_results_{method}_{gpt_version}_keyword.json'), 'r') as f:
                result_dict = json.load(f)
        except:
            result_dict = {}

        with open(os.path.join(root, f'(1){split}_results_{method}.json'), 'r') as f:
            metadata = json.load(f)
        with open(os.path.join(root, f'pathvqa_keyword.json'), 'r') as f:
            keyword = json.load(f)

        count_all = 0
        count_true = 0
        for qid, f in metadata.items():
            if qid in result_dict.keys():continue
            top_10_preds = f['top_10_preds']
            top_10_probs = f['top_10_probs']

            top_10_preds_kw = keyword[qid]['top_10_preds']
            top_10_probs_kw = keyword[qid]['top_10_probs']

            candidates = ""
            for i in range(5):
                candidates += f"{top_10_preds[i]}({round(float(top_10_probs[i]), 3)}), "

            keywords = ""
            for i in range(5):
                keywords += f"{top_10_preds_kw[i]}({round(float(top_10_probs_kw[i]), 3)}), "

            user_input = f"Keywords: {keywords}\n" + \
                         f"Question: {f['question']}\n" + \
                         f"Candidates: {candidates}\n" + \
                          "Answer: "
            messages = [sys_message]
            messages.append({"role": "user", "content": user_input})
            result, error = gpt_35_api_stream(messages)
            if not result:
                print(f"Error: {error}")
                time.sleep(5)
                continue

            prediction = messages[-1]['content']
            f['prediction'] = prediction
            if f["answer"] in prediction:
                f['result'] = True
            elif prediction in f["answer"]:
                f['result'] = True

            result_dict[qid] = f
            print(f"{user_input}\tAnswer: {f['answer']}\tAI response: {messages[-1]['content']}")
            if f['result'] == True:
                count_true += 1
            count_all += 1
            print(f'Accuracy: {count_true/count_all}')

            with open(os.path.join(root, f'(3){split}_results_{method}_{gpt_version}_keyword.json'), 'w') as f:
                json.dump(result_dict, f, indent=4)

