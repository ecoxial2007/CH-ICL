import torch
from torch import nn
import torch.nn.functional as F
import json

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores
# tao = nn.Parameter(torch.Tensor([0.04])).cuda()





def downstream_task_forward_topk(model, batch,  criterion, args, dataset=None):
    """
    Example simple function for performing forward pass over a batch input, obtaining predictions and a similarity loss.
    Modify to fit your specific task use case.
    """

    logits, image_features, query_type = model(batch)
    top_10_probs, top_10_preds = logits.topk(10, dim=1)



    top_10_preds = top_10_preds.cpu().numpy()
    top_10_probs = F.sigmoid(top_10_probs).cpu().numpy()
    try:
        with open(f'{args.dataset}_results.json', 'r') as f:
            results = json.load(f)
    except:
        results = {}

    if args.visible:
        vids, quess, qids, anss = batch['additional_info']


        for i, (vid, ques, preds, probs, ans, qid) in enumerate(zip(vids, quess, top_10_preds, top_10_probs, anss, qids)):
            preds = [dataset.label2keyword[pred] for pred in preds]
            probs = [float(round(prob, 4)) for prob in probs]  # round to 4 decimal places

            results[int(qid)] = {"img_id": vid, "question": ques,
                                 "top_10_preds": preds,
                                 "top_10_probs": probs}
            print(results[int(qid)])
    # Save results to json file
    with open(f'{args.dataset}_keyword_results.json', 'w') as f:
        json.dump(results, f, indent=4)
