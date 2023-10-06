import functools

import torch
import torch.nn.functional as F
import torch.nn as nn
import tqdm
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy
from .dist_utils import all_gather
from .gather import GatherLayer

import numpy as np

def cal_vqagm(pl_module, infer_unmask, vqa_targets, vqa_labels):
    infer = infer_unmask
    # gradient modulation
    t = infer["multi_modal_text_cls_feats"]
    i = infer["multi_modal_image_cls_feats"]

    hs = pl_module.hparams.config["hidden_size"]
    out_t = (torch.mm(t, torch.transpose(pl_module.vqa_head[-1].weight[:, :hs], 0, 1)) +
                pl_module.vqa_head[-1].bias / 2)
    out_i = (torch.mm(i, torch.transpose(pl_module.vqa_head[-1].weight[:, hs:], 0, 1)) +
                pl_module.vqa_head[-1].bias / 2)

    # print('out_t:', out_t.shape)
    # print('out_i:', out_i.shape)
    # print('vqa_targets:', vqa_targets.shape)

    vqa_loss_t = (F.binary_cross_entropy_with_logits(out_t, vqa_targets) * vqa_targets.shape[1])

    vqa_loss_i = (F.binary_cross_entropy_with_logits(out_i, vqa_targets) * vqa_targets.shape[1])

    # Modulation starts here !
    score_t = sum([F.softmax(out_t)[i][vqa_labels[i]] for i in range(out_t.size(0))])
    score_i = sum([F.softmax(out_i)[i][vqa_labels[i]] for i in range(out_i.size(0))])

    ratio_t = score_t / score_i
    ratio_i = 1 / ratio_t

    tanh = nn.Tanh()
    relu = nn.ReLU(inplace=True)

    if ratio_t > 1:
        coeff_t = 1 - tanh(0.1 * relu(ratio_t))
        coeff_i = 1
    else:
        coeff_i = 1 - tanh(0.1 * relu(ratio_i))
        coeff_t = 1

    # Using GradientModulation Callbacks to update paramaters' gradients:
    pl_module.coeff_t = coeff_t
    pl_module.coeff_i = coeff_i


def cal_clsgm(pl_module, infer_unmask, cls_labels):
    infer = infer_unmask
    # gradient modulation
    t = infer["multi_modal_text_cls_feats"]
    i = infer["multi_modal_image_cls_feats"]

    hs = pl_module.hparams.config["hidden_size"]
    out_t = (torch.mm(t, torch.transpose(pl_module.cls_head[3].weight[:, :hs], 0, 1)) +
                pl_module.cls_head[3].bias / 2)
    out_i = (torch.mm(i, torch.transpose(pl_module.cls_head[3].weight[:, hs:], 0, 1)) +
                pl_module.cls_head[3].bias / 2)

    cls_loss_t = F.cross_entropy(out_t, cls_labels)

    cls_loss_i = F.cross_entropy(out_i, cls_labels)

    # Modulation starts here !
    score_t = sum([F.softmax(out_t)[i][cls_labels[i]] for i in range(out_t.size(0))])
    score_i = sum([F.softmax(out_i)[i][cls_labels[i]] for i in range(out_i.size(0))])

    ratio_t = score_t / score_i
    ratio_i = 1 / ratio_t

    tanh = nn.Tanh()
    relu = nn.ReLU(inplace=True)

    if ratio_t > 1:
        coeff_t = 1 - tanh(0.1 * relu(ratio_t))
        coeff_i = 1
    else:
        coeff_i = 1 - tanh(0.1 * relu(ratio_i))
        coeff_t = 1

    # Using GradientModulation Callbacks to update paramaters' gradients:
    pl_module.coeff_t = coeff_t
    pl_module.coeff_i = coeff_i


def cal_irtrgm(pl_module, irtr_infer, false_len, _bs):
    infer = irtr_infer
    # gradient modulation
    t = infer["multi_modal_text_cls_feats"]
    i = infer["multi_modal_image_cls_feats"]

    hs = pl_module.hparams.config["hidden_size"]
    out_t = (torch.mm(t, torch.transpose(pl_module.irtr_head.weight[:, :hs], 0, 1)) +
                pl_module.irtr_head.bias / 2)
    out_i = (torch.mm(i, torch.transpose(pl_module.irtr_head.weight[:, hs:], 0, 1)) +
                pl_module.irtr_head.bias / 2)

    out_t = out_t[:, 0]
    out_t = rearrange(out_t, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    out_i = out_i[:, 0]
    out_i = rearrange(out_i, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)

    answer_t = torch.zeros(_bs).to(out_t).long()
    irtr_loss_t = F.cross_entropy(out_t, answer_t)

    answer_i = torch.zeros(_bs).to(out_i).long()
    irtr_loss_i = F.cross_entropy(out_i, answer_i)

    # Modulation starts here !
    score_t = sum([F.softmax(out_t)[i][answer_t[i]] for i in range(out_t.size(0))])
    score_i = sum([F.softmax(out_i)[i][answer_i[i]] for i in range(out_i.size(0))])

    ratio_t = score_t / score_i
    ratio_i = 1 / ratio_t

    tanh = nn.Tanh()
    relu = nn.ReLU(inplace=True)

    if ratio_t > 1:
        coeff_t = 1 - tanh(0.1 * relu(ratio_t))
        coeff_i = 1
    else:
        coeff_i = 1 - tanh(0.1 * relu(ratio_i))
        coeff_t = 1

    # Using GradientModulation Callbacks to update paramaters' gradients:
    pl_module.coeff_t = coeff_t
    pl_module.coeff_i = coeff_i

#-------------------------------------------------------------------------------------------------

def compute_feaclsl(pl_module, infer_masks, infer_unmask):
    infer = infer_masks
    # for text
    feaclsl_logits = infer["multi_modal_text_cls_feats"]
    feaclsl_labels = infer_unmask["multi_modal_text_cls_feats"]

    feaclsl_loss = (feaclsl_logits - feaclsl_labels) ** 2
    feaclsl_loss = torch.mean(feaclsl_loss)  # [N, L], mean loss per patch

    feaclsl_loss = feaclsl_loss * 10
    print('feaclsl_loss:', feaclsl_loss)

    ret = {
        "feaclsl_loss": feaclsl_loss,
        "feaclsl_logits": feaclsl_logits,
        "feaclsl_labels": feaclsl_labels,
    }


    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_feaclsl_loss")(ret["feaclsl_loss"])
    acc = -loss
    pl_module.log(f"feaclsl/{phase}/loss", loss)
    pl_module.log(f"feaclsl/{phase}/accuracy", acc)

    return ret

def compute_feaclsi(pl_module, infer_masks, infer_unmask):
    infer = infer_masks
    # for image

    feaclsi_logits = infer["multi_modal_image_cls_feats"]
    feaclsi_labels = infer_unmask["multi_modal_image_cls_feats"]

    feaclsi_loss = (feaclsi_logits - feaclsi_labels) ** 2
    feaclsi_loss = torch.mean(feaclsi_loss)  # [N, L], mean loss per patch

    feaclsi_loss = feaclsi_loss * 10
    print('feaclsi_loss:', feaclsi_loss)

    ret = {
        "feaclsi_loss": feaclsi_loss,
        "feaclsi_logits": feaclsi_logits,
        "feaclsi_labels": feaclsi_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_feaclsi_loss")(ret["feaclsi_loss"])
    acc = -loss
    pl_module.log(f"feaclsi/{phase}/loss", loss)
    pl_module.log(f"feaclsi/{phase}/accuracy", acc)

    return ret

def compute_feal(pl_module, infer_masks, infer_unmask):
    # 2. for feature token constraction
    # infer_unmask = pl_module.infer(batch, mask_text=False, mask_image=False)
    # infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    infer = infer_masks

    # for language
    feal_logits = infer["multi_modal_text_feats"]
    feal_labels = infer_unmask["multi_modal_text_feats"]

    feal_loss = (feal_logits - feal_labels) ** 2
    feal_loss = torch.mean(feal_loss)  # [N, L], mean loss per patch

    feal_loss = feal_loss * 10
    print('feal_loss:', feal_loss)

    ret = {
        "feal_loss": feal_loss,
        "feal_logits": feal_logits,
        "feal_labels": feal_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_feal_loss")(ret["feal_loss"])
    acc = -loss
    pl_module.log(f"feal/{phase}/loss", loss)
    pl_module.log(f"feal/{phase}/accuracy", acc)

    return ret

def compute_feai(pl_module, infer_masks, infer_unmask):
    # 2. for feature token constraction
    # infer_unmask = pl_module.infer(batch, mask_text=False, mask_image=False)
    # infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    infer = infer_masks

    # for image
    ## for infer
    if pl_module.hparams.config["mim_layer"] == -1:
        multi_modal_image_feats = infer["multi_modal_image_feats"]
    else:
        layer_idx = pl_module.hparams.config["mim_layer"]
        multi_modal_image_feats = infer[f"multi_modal_image_feats_{layer_idx}"]

    x = pl_module.mim_head.decoder_embed(multi_modal_image_feats)
    ids_restore = infer["mim_ids_restore"]
    # append mask tokens to sequence
    mask_tokens = pl_module.mim_head.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    # remove cls token
    x = x[:, 1:, :]
    feai_logits = x
    # print(feai_logits.shape) # 

    ## for infer_unmask
    unmask_multi_modal_image_feats = infer_unmask["multi_modal_image_feats"]
    unmask_multi_modal_image_feats = pl_module.mim_head.decoder_embed(unmask_multi_modal_image_feats)
    # remove cls token
    unmask_multi_modal_image_feats = unmask_multi_modal_image_feats[:, 1:, :]
    feai_labels = unmask_multi_modal_image_feats
    # print(feai_labels.shape) # torch.Size([16, 324, 384]) torch.Size([16, 324, 384]) torch.Size([16, 324])


    mask = infer["mim_masks"]
    # print(mask.shape)
    feai_loss = (feai_logits - feai_labels) ** 2
    feai_loss = feai_loss.mean(dim=-1)  # [N, L], mean loss per patch
    feai_loss = (feai_loss * mask).sum() / mask.sum()  # mean loss on removed patches

    feai_loss = feai_loss * 10
    print('feai_loss:', feai_loss)

    ret = {
        "feai_loss": feai_loss,
        "feai_logits": feai_logits,
        "feai_labels": feai_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_feai_loss")(ret["feai_loss"])
    acc = -loss
    pl_module.log(f"feai/{phase}/loss", loss)
    pl_module.log(f"feai/{phase}/accuracy", acc)

    return ret


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor):
    text_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (text_loss + image_loss) / 2.0

def clip_metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2text_match_idx = similarity.argmax(dim=1)
    text2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2text_match_idx == y).float().mean()
    text_acc = (text2img_match_idx == y).float().mean()

    clip_acc = img_acc + text_acc

    return clip_acc

def compute_itc(pl_module, infer_masks):
    # images, text = batch
    text_mask = infer_masks["multi_modal_text_cls_feats"]
    image_mask = infer_masks["multi_modal_image_cls_feats"]

    text_mask = torch.cat(GatherLayer.apply(text_mask), dim=0)
    image_mask = torch.cat(GatherLayer.apply(image_mask), dim=0)

    # print('shape of text and image:', text_mask.shape, image_mask.shape)
    # shape of cls text and image: torch.Size([270, 768]) torch.Size([270, 768])
    # shape of all text and image: torch.Size([270, 64, 768]) torch.Size([270, 82, 768])

    # text_dev = {k: v.to(pl_module.device) for k, v in pl_module.tokenizer(text).items()}

    # image_embed = pl_module.vision_encoder(images)
    # text_embed = pl_module.text_encoder(text)
    # similarity = text_embed @ image_embed.T
    
    # N = text_mask.shape[0] 
    
    # image_embed = pl_module.clip_projection(image_mask)
    # text_embed = pl_module.clip_projection(text_mask)
    # similarity = text_embed @ image_embed.T

    similarity = text_mask @ image_mask.T

    # similarity = text_embed @ image_embed.transpose(1,2)

    itc_loss = clip_loss(similarity)
    itc_acc = clip_metrics(similarity)

    print('itc loss:', itc_loss)

    ret = {
        "itc_loss": itc_loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itc_loss")(ret["itc_loss"])
    acc = -loss
    pl_module.log(f"itc/{phase}/loss", loss)
    pl_module.log(f"itc/{phase}/accuracy", acc)

    return ret


def compute_bb(pl_module, infer_masks):

    # text_unmask = infer_unmask["multi_modal_text_feats"]
    # image_unmask = infer_unmask["multi_modal_image_feats"]
    # text_mask = infer_masks["multi_modal_text_feats"]
    # image_mask = infer_masks["multi_modal_image_feats"]

    text_mask = infer_masks["multi_modal_text_cls_feats"]
    image_mask = infer_masks["multi_modal_image_cls_feats"]

    text_mask = torch.cat(GatherLayer.apply(text_mask), dim=0)
    image_mask = torch.cat(GatherLayer.apply(image_mask), dim=0)
    # print(text_unmask.shape) [16, 64, 768]
    # print(image_unmask.shape) [16, 325, 768]
    # print(text_mask.shape) [16, 64, 768]
    # print(image_mask.shape) [16, 82, 768]
    
    # number of samples
    N = text_mask.shape[0] 
    # print('batch size of all progress is N:', N)
    # text_unmask = text_unmask.view(N, -1)
    # print('text_unmask:', text_unmask)
    # text_mask = text_mask.view(N, -1)
    # print('text_mask:', text_mask)

    # image_unmask = image_unmask.view(N, -1)
    # print('image_unmask:', image_unmask)
    # image_mask = image_mask.view(N, -1)
    # print('image_mask:', image_mask)


    # image_unmask_sim = image_unmask.mm(image_unmask.t())
    # print('1:',image_unmask_sim)
    # image_unmask_norm = torch.norm(image_unmask_sim, 2, 1).view(-1, 1)
    # print('2', image_unmask_norm)
    # image_unmask_sim = image_unmask_sim / image_unmask_norm
    # print('3image_unmask_sim:', image_unmask_sim)
    # text_unmask_sim = text_unmask.mm(text_unmask.t()) 
    # text_unmask_norm = torch.norm(text_unmask_sim, 2, 1).view(-1, 1)
    # text_unmask_sim = text_unmask_sim / text_unmask_norm
    # print('text_unmask_sim:', text_unmask_sim)
    # batch_loss_unmask = (image_unmask_sim - text_unmask_sim) ** 2 / N
    # batch_loss_unmask = torch.sum(batch_loss_unmask)
    # print('batch_loss_unmask:', batch_loss_unmask)

    image_mask_sim = image_mask.mm(image_mask.t())
    image_mask_norm = torch.norm(image_mask_sim, 2, 1).view(-1, 1)
    image_mask_sim = image_mask_sim / image_mask_norm
    # print('image_mask_sim:', image_mask_sim)

    text_mask_sim = text_mask.mm(text_mask.t()) 
    text_mask_norm = torch.norm(text_mask_sim, 2, 1).view(-1, 1)
    text_mask_sim = text_mask_sim / text_mask_norm
    # print('text_mask_sim:', text_mask_sim)

    batch_loss_mask = (image_mask_sim - text_mask_sim) ** 2 / N
    batch_loss_mask = torch.sum(batch_loss_mask)

    batch_loss_mask = batch_loss_mask * 3000
    print('bb_loss:', batch_loss_mask)

    # batch_loss = (batch_loss_unmask + batch_loss_mask) / 2
    # print('bb_loss:', batch_loss.data)

    ret = {
        "bb_loss": batch_loss_mask,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_bb_loss")(ret["bb_loss"])
    acc = -loss
    pl_module.log(f"bb/{phase}/loss", loss)
    pl_module.log(f"bb/{phase}/accuracy", acc)

    return ret


def compute_mlm(pl_module, batch, infer_masks):
    # infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    infer = infer_masks
    mlm_logits = pl_module.mlm_head(infer["multi_modal_text_feats"])
    mlm_labels = infer["text_labels"]

    # print('mlm_logits:', mlm_logits.shape)
    # print('mlm_labels:', mlm_labels.shape)
    # print('vocal_size:', pl_module.hparams.config["vocab_size"])
    # mlm_logits: torch.Size([16, 32, 50265])
    # mlm_labels: torch.Size([16, 32])
    # vocal_size: 50265

    # method 1: direct CE
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    '''
    # method 2: weighted CE by multiply weight_matrix calculated mean channel self attention map
    with torch.no_grad():
        channel_weights = torch.mean(mlm_logits, dim=2)
        Q = channel_weights.clone().detach().requires_grad_(False).unsqueeze(-1).to(mlm_logits.device) 
        K = channel_weights.clone().detach().requires_grad_(False).unsqueeze(-1).to(mlm_logits.device)
        V = channel_weights.clone().detach().requires_grad_(False).unsqueeze(-1).to(mlm_logits.device)

        k_dim = K.shape[-1]
        attn_logits = Q @ K.transpose(1, 2) / torch.sqrt(torch.tensor(k_dim, dtype=torch.float32))
        attention_weights = torch.softmax(attn_logits, dim=-1) #(N, L, L)
        attention_output = attention_weights @ V #(N, L, C)
        mlm_logits = mlm_logits * attention_output
    
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )
    '''
    # weight_matrix = channel_weights.clone().detach().requires_grad_(False).to(mlm_logits.device)
    # weight_matrix = weight_matrix / torch.sum(weight_matrix)
    # mlm_loss = F.cross_entropy(
        # mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        # mlm_labels.view(-1),
        # ignore_index=-100,
        # weight=weight_matrix
    # )

    # print('mlm_loss:', mlm_loss)

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    # print(mlm_loss.shape, mlm_logits.shape) #torch.Size([]) torch.Size([16, 64, 50265])
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(ret["mlm_logits"], ret["mlm_labels"])
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def compute_mim(pl_module, batch, infer_masks):
    # infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    infer = infer_masks

    if pl_module.hparams.config["mim_layer"] == -1:
        multi_modal_image_feats = infer["multi_modal_image_feats"]
    else:
        layer_idx = pl_module.hparams.config["mim_layer"]
        multi_modal_image_feats = infer[f"multi_modal_image_feats_{layer_idx}"]

    mim_logits = pl_module.mim_head(multi_modal_image_feats, infer["mim_ids_restore"])

    target = infer["patched_images"]
    if pl_module.hparams.config["norm_pix_loss"]:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
    mim_labels = target # [N,P,C]
    # print('1label', mim_labels.shape)
    mask = infer["mim_masks"] # [N,P]
    # print('2mask', mask.shape) #torch.Size([16, 324, 768]) torch.Size([16, 324]) torch.Size([16, 324])

    '''
    # method 2: weighted MSE by multiply weight_matrix calculated mean channel 
    channel_weights = torch.mean(mim_logits, dim=2) # [N,P] for dim=2 keep BATCH PATCH; [N] for dim=(1, 2) keep dimension 1 .unsqueeze(1).unsqueeze(-1).
    weight_matrix = channel_weights.clone().detach().requires_grad_(False).unsqueeze(-1).to(mim_logits.device) # [N,P,1] for dim=2, [N,1,1] for dim=(1,2)

    weight_matrix = weight_matrix / torch.sum(weight_matrix)
    mim_loss = ((mim_logits - mim_labels) ** 2) * weight_matrix # [N,P,C] * [N,1,1]
    '''

    # method 1: direct MSE 
    mim_loss = (mim_logits - mim_labels) ** 2 # [N,P,C]

    mim_loss = mim_loss.mean(dim=-1)  # # [N,P] mean loss per patch
    mim_loss = (mim_loss * mask).sum() / mask.sum()  # A NUMBER mean loss on removed patches 

    # 1label torch.Size([16, 324, 768])
    # 2mask torch.Size([16, 324])
    # 3mim_loss: torch.Size([16, 324, 768])
    # 4mim_loss:, torch.Size([16, 324])
    # 5mim_loss: tensor(0.7575, device='cuda:0')
    # channel weights torch.Size([16])
    # weight_matrix torch.Size([16, 1, 1])
    # channel weights torch.Size([16, 324])
    # weight_matrix torch.Size([16, 324, 1])

    ret = {
        "mim_loss": mim_loss,
        "mim_logits": mim_logits,
        "mim_labels": mim_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mim_loss")(ret["mim_loss"])
    acc = -loss
    pl_module.log(f"mim/{phase}/loss", loss)
    pl_module.log(f"mim/{phase}/accuracy", acc)

    return ret

def compute_mlm_preteacher(pl_module, batch, infer_masks, infer_masks_preteacher):
    # infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    # for student
    infer = infer_masks
    mlm_logits = pl_module.mlm_head(infer["multi_modal_text_feats"])

    # for preteacher
    # with torch.no_grad():
    # preteacherl_logits = pl_module.preteacher.module.mlm_head(infer_masks_preteacher["multi_modal_text_feats"])
    preteacherl_logits = pl_module.mlm_head(infer_masks_preteacher["multi_modal_text_feats"])

    preteacherl_loss = (preteacherl_logits - mlm_logits) ** 2
    preteacherl_loss = torch.mean(preteacherl_loss)  # [N, L], mean loss per patch

    # print('preteacherl_loss:', preteacherl_loss)

    # prel_loss = preteacherl_loss + mlm_loss
    prel_loss = preteacherl_loss

    ret = {
        "prel_loss": prel_loss,
    }

    phase = "train" if pl_module.training else "val"
    # print(mlm_loss.shape, mlm_logits.shape) #torch.Size([]) torch.Size([16, 64, 50265])
    loss = getattr(pl_module, f"{phase}_prel_loss")(ret["prel_loss"])
    acc = -loss
    pl_module.log(f"prel/{phase}/loss", loss)
    pl_module.log(f"prel/{phase}/accuracy", acc)

    return ret

def compute_mim_preteacher(pl_module, batch, infer_masks, infer_masks_preteacher):
    # infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    infer = infer_masks
    # for student
    if pl_module.hparams.config["mim_layer"] == -1:
        multi_modal_image_feats = infer["multi_modal_image_feats"]
    else:
        layer_idx = pl_module.hparams.config["mim_layer"]
        multi_modal_image_feats = infer[f"multi_modal_image_feats_{layer_idx}"]

    mim_logits = pl_module.mim_head(multi_modal_image_feats, infer["mim_ids_restore"])

    mask = infer["mim_masks"] # [N,P]

    # for preteacher
    # with torch.no_grad():
    if pl_module.hparams.config["mim_layer"] == -1:
        preteacheri_multi_modal_image_feats = infer_masks_preteacher["multi_modal_image_feats"]
    else:
        layer_idx = pl_module.hparams.config["mim_layer"]
        preteacheri_multi_modal_image_feats = infer_masks_preteacher[f"multi_modal_image_feats_{layer_idx}"]

    # preteacheri_multi_modal_image_feats = infer_masks_preteacher["multi_modal_image_feats"]

    preteacheri_logits = pl_module.preteacher.module.mim_head(preteacheri_multi_modal_image_feats, infer_masks_preteacher["mim_ids_restore"])
    # preteacheri_logits = pl_module.mim_head(preteacheri_multi_modal_image_feats, infer_masks_preteacher["mim_ids_restore"])

    preteacheri_loss = (preteacheri_logits - mim_logits) ** 2
    preteacheri_loss = preteacheri_loss.mean(dim=-1)
    preteacheri_loss = (preteacheri_loss * mask).sum() / mask.sum() 

    # print('preteacheri_loss:', preteacheri_loss)

    # prei_loss = preteacheri_loss + mim_loss
    prei_loss = preteacheri_loss

    ret = {
        "prei_loss": prei_loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_prei_loss")(ret["prei_loss"])
    acc = -loss
    pl_module.log(f"prei/{phase}/loss", loss)
    pl_module.log(f"prei/{phase}/accuracy", acc)

    return ret


def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(pl_module.device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    # infer = pl_module.infer(batch, mask_text=True, mask_image=True)

    itm_logits = pl_module.itm_head(infer["multi_modal_cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    itm_loss = itm_loss * 10
    print('itm_loss:', itm_loss)

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_vqa(pl_module, batch, infer_unmask, test=False):
    # infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    infer = infer_unmask
    vqa_logits = pl_module.vqa_head(infer["multi_modal_cls_feats"])
    vqa_targets = torch.zeros(len(vqa_logits), pl_module.hparams.config["vqa_label_size"]).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]
    vqa_answer_types = torch.tensor(batch["answer_types"]).to(pl_module.device)

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    # print('vqa_logts.shape:', vqa_logits.shape)
    vqa_loss = (F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1])

    # apply gradient modulation module
    cal_vqagm(pl_module, infer_unmask, vqa_targets, vqa_labels)

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
        "vqa_answer_types": vqa_answer_types,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(ret["vqa_logits"], ret["vqa_targets"], ret["vqa_answer_types"])
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_cls(pl_module, batch, infer_unmask, test=False):
    # infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    infer = infer_unmask

    cls_logits = pl_module.cls_head(infer["multi_modal_cls_feats"])
    cls_labels = batch["cls_labels"]
    cls_loss = F.cross_entropy(cls_logits, cls_labels)

    # apply gradient modulation module
    cal_clsgm(pl_module, infer_unmask, cls_labels)

    ret = {
        "cls_loss": cls_loss,
        "cls_logits": cls_logits,
        "cls_labels": cls_labels,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_cls_loss")(ret["cls_loss"])
    acc = getattr(pl_module, f"{phase}_cls_accuracy")(ret["cls_logits"], ret["cls_labels"])
    evallscore = getattr(pl_module, f"{phase}_cls_evallscore")(ret["cls_logits"], ret["cls_labels"])

    pl_module.log(f"cls/{phase}/loss", loss)
    pl_module.log(f"cls/{phase}/accuracy", acc)
    pl_module.log(f"cls/{phase}/evallscore", evallscore)

    return ret


def compute_irtr(pl_module, batch, test=False):
    is_training_phase = pl_module.training
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack([batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1)
    text_masks = torch.stack([batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1)
    text_labels = torch.stack([batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1)

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    batch_infer = {
        "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
        "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
        "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
        "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
    }

    ## original m3ae irtr
    infer = pl_module.infer(batch_infer)
    
    ## double masked version
    # infer = pl_module.infer(batch_infer, mask_text=False, mask_image=True)

    score = pl_module.irtr_head(infer["multi_modal_cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    # apply gradient modulation module
    cal_irtrgm(pl_module, infer, false_len, _bs)

    ret = {"irtr_loss": irtr_loss}

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=256,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(text_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(image_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    # TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        # == Begin: Add New Keys ==
        batch_text_preload = {
            "text_ids": _b["text_ids"].to(pl_module.device),
            "text_masks": _b["text_masks"].to(pl_module.device),
            "text_labels": _b["text_labels"].to(pl_module.device),
            "img_index": _b["img_index"],
        }
        text_preload.append(batch_text_preload)
        # == End  : Add New Keys ==

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                # == Begin: Add New Keys ==
                batch_infer = {
                    "text_ids": txt_batch["text_ids"],
                    "text_masks": txt_batch["text_masks"],
                    "text_labels": txt_batch["text_labels"],
                }
                score = pl_module.irtr_head(pl_module.infer(batch_infer, img=im, )["multi_modal_cls_feats"])[:, 0]
                # == End  : Add New Keys ==

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)
