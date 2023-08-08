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

def downstream_task_forward(model, batch,  criterion, args, dataset=None):
    """
    Example simple function for performing forward pass over a batch input, obtaining predictions and a similarity loss.
    Modify to fit your specific task use case.
    """
    if args.dataset == 'peir':
        y_gt_bce = batch['labels_matrix']
        preds, image_features, query_type = model(batch)
        loss = criterion(preds, y_gt_bce)
        scores = compute_score_with_logits(preds, y_gt_bce)
        accs = torch.sum(scores, dim=1)

    else:
        x_txt_cands_mc = batch['text_cands_features']
        y_gt_mc = batch['labels_id']
        merge_features, image_features, query_type = model(batch)
        y_pred = F.cosine_similarity(merge_features.unsqueeze(1), x_txt_cands_mc, dim=-1)  # (N, N_ans)
        loss = criterion(y_pred / 0.04, y_gt_mc)
        accs = (y_pred.argmax(dim=-1) == y_gt_mc).float()

    if not args.visible:
        return loss, accs

    else:
        l_y_pred = list(y_pred.argmax(dim=-1).cpu().numpy())
        l_y_gt = list(y_gt_mc.cpu().numpy())
        vids, quess, anss, typee = batch['additional_info']

        for i, (vid, ques, i_pred, i_gt, type) in enumerate(zip(vids, quess, l_y_pred, l_y_gt, typee)):
            try:
                line = f"{vid}\t{ques}\t{anss[i_pred][i]}\t{anss[i_gt][i]}\t{i_gt==i_pred}\t{type}"
            except:
                line = f"{vid}\t{ques}\t{dataset.label2answer[i_pred]}\t{dataset.label2answer[i_gt]}\t{i_gt == i_pred}\t{type}"
            print(line)
            with open(f'{args.dataset}_{model.config.split}.csv', 'a') as f:
                f.write(line+'\n')



def downstream_task_forward_topk(model, batch,  criterion, args, dataset=None):
    """
    Example simple function for performing forward pass over a batch input, obtaining predictions and a similarity loss.
    Modify to fit your specific task use case.
    """

    x_txt_cands_mc = batch['text_cands_features']
    y_gt_mc = batch['labels_id']

    merge_features, image_features, query_type = model(batch)
    y_pred  = F.cosine_similarity(merge_features.unsqueeze(1), x_txt_cands_mc, dim=-1)  # (N, N_ans)

    top_10_preds = torch.topk(y_pred, 10, dim=-1).indices.cpu().numpy()
    top_10_probs = torch.topk(F.softmax(y_pred/ 0.04, dim=-1), 10, dim=-1).values.cpu().numpy()

    loss    = criterion(y_pred / 0.04, y_gt_mc)
    accs    = (y_pred.argmax(dim=-1) == y_gt_mc).float()

    try:
        with open(f'{args.dataset}_results.json', 'r') as f:
            results = json.load(f)
    except:
        results = {}

    if args.visible:
        vids, quess, qids, anss = batch['additional_info']
        atype = batch['atype']

        for i, (vid, ques, preds, probs, ans, qid) in enumerate(zip(vids, quess, top_10_preds, top_10_probs, anss, qids)):
            preds = [dataset.label2answer[pred] for pred in preds]
            probs = [float(round(prob, 4)) for prob in probs]  # round to 4 decimal places
            if preds[0] in ans or preds[0] in ans:
                result = True
            else:
                result = False
            results[int(qid)] = {"img_id": vid, "question": ques,
                                 "top_10_preds": preds,
                                 "top_10_probs": probs,
                                 "atype": atype[i],
                                 "answer": ans,
                                 "prediction": preds[0],
                                 "result": result}

    # Save results to json file
    with open(f'{args.dataset}_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return loss, accs