import time
import torch
import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    set_model_global_prune,
    set_model_global_threshold
)

__all__ = ["train", "validate", "modifier"]

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()
def get_mask(sparsity,model,args,threshold=None):
        if args.global_prune:
            local=[]
            for name, p in model.named_parameters():
                if hasattr(p, 'is_score') and p.is_score:
                    #threshold=percentile(p,sparsity*100)
                    mask=p.detach()<threshold
                    local.append(mask.detach().flatten())
            local=torch.cat(local)
        else:
            local=[]
            for name, p in model.named_parameters():
                if hasattr(p, 'is_score') and p.is_score:
                    threshold=percentile(p,sparsity*100)
                    mask=p.detach()<threshold
                    local.append(mask.detach().flatten())
            local=torch.cat(local)

        
        """
        print("sparsity",sparsity,"threshold",threshold)
        total_n=0.0
        total_re=0.0
        for name, p in self.named_parameters():
            if hasattr(p, 'is_score') and p.is_score:
                mask=p.detach()<threshold
                mask=mask.float()
                total_re+=mask.sum().item()
                total_n+=mask.numel()
                print(name,":masked ratio",mask.sum().item()/mask.numel())
        print("total remove",total_re/total_n)
        """
        return local  
def get_threshold(model,sparsity):
    local=[]
    for name, p in model.named_parameters():
        if hasattr(p, 'is_score') and p.is_score:
            local.append(p.detach().flatten())
    #print("num of pruning params:",len(local))
    local=torch.cat(local)
    threshold=percentile(local,sparsity*100)
    return threshold   

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    loss_changed=AverageMeter("LossGap",":.5f")
    mask_changed=AverageMeter("MaskChangeRate",":.5f")
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5,mask_changed,loss_changed],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    mask_perturb_rate=0.0
    loss_perturb=0.0
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        threshod=None
        ##############
        if args.global_prune:
            threshod=get_threshold(model,args.prune_rate)
            set_model_global_threshold(model,threshod)
        # compute output
        pre_mask=get_mask(args.prune_rate,model,args,threshod)

        output = model(images)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        ##############
        if args.global_prune:
            threshod=get_threshold(model,args.prune_rate)
            set_model_global_threshold(model,threshod)

        after_mask=get_mask(args.prune_rate,model,args,threshod)
        
        loss1=criterion(model(images), target)
        loss1.backward()
        optimizer.second_step(zero_grad=True)

        mask_perturb_rate=((after_mask+pre_mask).sum().detach().float().item()/pre_mask.sum().detach().float().item()-1)
        loss_perturb=(loss1-loss).detach().item()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        loss_changed.update(loss_perturb,1)
        mask_changed.update(mask_perturb_rate,1)


        # compute gradient and do SGD step
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if (i+1) % args.print_freq == 0:
    t = (num_batches * epoch + i) * batch_size
    progress.display(i)
    progress.write_to_tensorboard(writer, prefix="train", global_step=t)
    #print("mask_perturb_rate",mask_perturb_rate/(i+1),"loss_perturb",loss_perturb/(i+1))
    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if i % args.print_freq == 0:
        progress.display(i)
        #progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg

def modifier(args, epoch, model):
    return
