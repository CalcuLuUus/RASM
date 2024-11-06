import torch
import torch.nn as nn
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

# def load_checkpoint(model, weights):
#     checkpoint = torch.load(weights)
#     try:
#         model.load_state_dict(checkpoint["state_dict"])
#     except:
#         state_dict = checkpoint["state_dict"]
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:] if 'module.' in k else k
#             new_state_dict[name] = v
#         model.load_state_dict(new_state_dict)
def load_checkpoint(model, weights):
    print(f'weight format: {weights[-2:]}')
    if weights[-2:] == 'pt':
        checkpoint = torch.load(weights)
        try:
            # model.load_state_dict(checkpoint["state_dict"])
            model.load_state_dict(checkpoint["model"])
        except:
            # state_dict = checkpoint["state_dict"]
            state_dict = checkpoint["model"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if 'module.' in k else k
                new_state_dict[name] = v
            tmp = model.load_state_dict(new_state_dict, strict=False)
            print(tmp)
    else:
        checkpoint = torch.load(weights)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if 'module.' in k else k
                new_state_dict[name] = v
            tmp = model.load_state_dict(new_state_dict, strict=True)
            print(tmp)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import RASM
    model_restoration = RASM(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_projection=opt.token_projection,token_mlp=opt.token_mlp)

    return model_restoration