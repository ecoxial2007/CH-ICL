import os
import sys
sys.path.append('./')
import glob
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from trainval_single_batch import downstream_task_forward
from loss import LabelSmoothingCrossEntropy

dataset_mapping = {
    'radvqa': 'src.datasets.medvqa_features',
    'pathvqa': 'src.datasets.medvqa_features',
    'slakevqa': 'src.datasets.medvqa_features',
    'peir': 'src.datasets.medvqa_peir'
}
# dataset_mapping = {
#     'radvqa': 'datasets.medvqa',
#     'pathvqa': 'datasets.medvqa',
#     'slakevqa': 'datasets.medvqa',
#     'peir': 'datasets.peir'
# }
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def log(message, logger=None):
    """
    Placeholder log function; replace with your loggers/apis of choice (e.g. wandb, etc.)
    """
    if logger is not None: raise NotImplemented("implement your own logger")
    print(message)
    if args.freeze:
        args.log_path = os.path.join("./checkpoints", f"{args.dataset}", args.method+'_freeze')
    else:
        args.log_path = os.path.join("./checkpoints", f"{args.dataset}", args.method + '_all')
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    with open(os.path.join(args.log_path,  'log.txt'), 'a') as lf:
        lf.write(message+'\n')

def process_batch(batch, set_to_device=None, replace_empty_with_none=False):
    if set_to_device is not None:
        if isinstance(batch, dict):
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(set_to_device)
        elif isinstance(batch, list):
            batch = [_b.to(set_to_device) if torch.is_tensor(_b) else _b for _b in batch]

    if replace_empty_with_none:
        if isinstance(batch, dict):
            for key in batch:
                if len(batch[key]) == 0:
                    batch[key] = None
        elif isinstance(batch, list):
            batch = [_b if len(_b) > 0 else None for _b in batch]

    return batch


def main(args):
    seed_everything(args.seed)
    if args.dataset == 'peir':
        from model_peir import LGVAConfig, LGVAModel
    else:
        from model_features import LGVAConfig, LGVAModel


    # create LGVAConfig from model hyperparameters
    config = LGVAConfig.from_args(args)
    device = torch.device("cuda")

    model = LGVAModel(config, device).to(device)
    # Assuming model.backbone exists and you want to freeze it
    try:
        if args.method == 'biomed' and args.freeze:
            for param in model.backbone.parameters():
                param.requires_grad = False
    except:
        pass

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters: ", total_params)
    print("Trainable parameters: ", trainable_params)

    if args.dataset in dataset_mapping:
        module_name = dataset_mapping[args.dataset]
        ImageLanguageDataset = getattr(__import__(module_name, fromlist=['ImageLanguageDataset']),
                                       'ImageLanguageDataset')
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    dset_train = ImageLanguageDataset(args, split="train")
    dldr_train = torch.utils.data.DataLoader(dset_train,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=False,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)
    if args.dataset != 'peir':
        dset_val = ImageLanguageDataset(args, split="val")
        dldr_val = torch.utils.data.DataLoader(dset_val,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               pin_memory=False,
                                               drop_last=False,
                                               num_workers=args.num_workers,
                                               collate_fn=default_collate)
    warmup_epochs = int(0.1*args.epochs)
    # create optimizer
    # create optimizer
    if args.wd > 0.0:
        optim = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,  # peak learning rate
                                  betas=(0.9, 0.98),  # optimizer momentum β1, β2
                                  eps=1.0e-6,  # eps
                                  weight_decay=args.wd)  # weight decay
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)  # peak learning rate

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs-warmup_epochs)
    # gamma = 0.95
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    warmup_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    scheduler_with_warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=[warmup_lambda])

    if args.dataset == 'peir':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        criterion = LabelSmoothingCrossEntropy()


    # simple training loop (for illustrative purposes)
    all_train_accs = []
    for epoch_i in range(args.epochs):
        # train epoch
        model.train()
        for i, batch in enumerate(dldr_train):
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
            # refactored the "forward pass" here
            loss, accs = downstream_task_forward(model, batch, criterion, args)
            all_train_accs.append(accs)
            model.zero_grad(set_to_none=True)
            loss.backward()

            # do logging stuff with accs, loss, etc. For example:
            log(f"train: epoch{epoch_i}, lr = {optim.param_groups[0]['lr']}, iter{i}: loss = {loss.item()}, acc = {accs.mean().item()}")
            if args.grad_clip_val > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_val)

            optim.step()
            optim.zero_grad()

            # Update the learning rate
            if epoch_i < warmup_epochs:
                scheduler_with_warmup.step()
            else:
                scheduler.step()

        torch.cuda.empty_cache()

        if args.dataset != 'peir':
            # val epoch
            model.eval()
            all_val_accs = []
            for i, batch in enumerate(dldr_val):
                with torch.no_grad():
                    batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
                    loss, accs = downstream_task_forward(model, batch, criterion, args)
                    all_val_accs.append(accs)

            overall_acc = torch.cat(all_val_accs).mean().item()
            log(f"val: epoch{epoch_i}: overall_acc = {overall_acc}")
        else:
            overall_acc = torch.cat(all_train_accs).mean().item()



        checkpoint = {
            "epoch": epoch_i,
            "overall_acc": overall_acc,
            "state_dict": model.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.log_path, f"ckpt_{overall_acc}.pth"))

        files = glob.glob(os.path.join(args.log_path, f'ckpt_*.pth'))
        sorted_files = sorted(files, key=lambda x: float(x.split('_')[-1].split('.pth')[0]))
        for filepath in sorted_files[:-1]:
            os.remove(filepath)
        if overall_acc > 0.801: quit()
    return




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parser for LGVA training script.")

    # Training hyperparameters
    parser.add_argument('--batch_size', default=256, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-3, type=float, help="learning rate")
    parser.add_argument('--wd', default=1e-2, type=float, help="weight decay")
    parser.add_argument('--epochs', default=50, type=int, help="number of training epochs")
    parser.add_argument('--grad_clip_val', default=1.0, type=float, help="gradient clip, must be set > 0 to enable")
    parser.add_argument('--gpus', default=1, type=int)  # NOTE: current script is set-up for single-gpu training only.
    parser.add_argument('--num_workers', default=1, type=int, help="number of dataset workers")
    parser.add_argument('--seed', default=3407, type=int, help="random seed")

    # Efficient model hyperparameters (for more help/details, see LGVAConfig)
    parser.add_argument('--d_input', default=768, type=int, help="see LGVAConfig")
    parser.add_argument('--n_ca_heads', default=12, type=int, help="see LGVAConfig")
    parser.add_argument('--ca_dropout', default=0.1, type=float, help="see LGVAConfig")


    # I/O and tools parameters
    parser.add_argument('--data_path', type=str, help='Annotation', default='./data/Annotation')
    parser.add_argument('--feature_path', type=str, help='Feature')
    parser.add_argument('--method', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--visible', action='store_true') #for check each question predicts answer
    args = parser.parse_args()

    main(args)
