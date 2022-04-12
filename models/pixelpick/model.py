import os
from math import ceil
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.losses import custom_losses
from utils.metrics import AverageMeter, RunningScore
from utils.utils import Visualiser, get_dataloader, get_model, get_optimizer, get_lr_scheduler, write_log
from query import QuerySelector


class Model:
    def __init__(self, args):
        # common args
        self.args = args
        self.best_loss_val = float('inf')
        self.best_loss_train = float('inf')
        self.dataset_name = args.dataset_name
        self.debug = args.debug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.experim_name = args.experim_name
        self.dir_checkpoints = args.dir_checkpoints
        self.ignore_index = args.ignore_index
        self.init_n_pixels = args.n_init_pixels
        self.max_budget = args.max_budget
        self.n_classes = args.n_classes
        self.n_epochs = args.n_epochs
        self.n_pixels_by_us = args.n_pixels_by_us
        self.network_name = args.network_name
        self.nth_query = -1
        self.loss = args.loss
        self.continuity_weight = args.continuity_weight

        self.dataloader = get_dataloader(deepcopy(args), val=False, query=False,
                                         shuffle=True, batch_size=args.batch_size, n_workers=args.n_workers)
        self.dataloader_query = get_dataloader(deepcopy(args), val=False, query=True,
                                               shuffle=False, batch_size=1, n_workers=args.n_workers)
        self.dataloader_val = get_dataloader(deepcopy(args), val=True, query=False,
                                             shuffle=False, batch_size=1, n_workers=args.n_workers)

        self.model = get_model(args).to(self.device)

        self.lr_scheduler_type = args.lr_scheduler_type
        self.query_selector = QuerySelector(args, self.dataloader_query)
        self.vis = Visualiser(args.dataset_name)
        # for tracking stats
        self.running_loss, self.running_score = AverageMeter(), RunningScore(args.n_classes)

        self.tb_writer = None
        self.epochs_no_improve = 0
        self.early_stopping_patience = 100
        # if active learning
        # if self.n_pixels_by_us > 0:
        #     self.model_0_query = f"{self.dir_checkpoints}/0_query_{args.seed}.pt"

    def __call__(self):
        # fully-supervised model
        if self.n_pixels_by_us == 0:
            dir_checkpoints = f"{self.dir_checkpoints}/fully_sup"
            os.makedirs(f"{dir_checkpoints}", exist_ok=True)

            self.log_train, self.log_val = f"{dir_checkpoints}/log_train.txt", f"{dir_checkpoints}/log_val.txt"
            write_log(f"{self.log_train}", header=["epoch", "mIoU", "pixel_acc", "loss"])
            write_log(f"{self.log_val}", header=["epoch", "mIoU", "pixel_acc"])
            self.tb_writer = SummaryWriter(log_dir=os.path.join(dir_checkpoints, 'logs'))

            self._train()

        # active learning model
        else:
            n_stages = self.max_budget // self.n_pixels_by_us
            n_stages += 1 if self.init_n_pixels > 0 else 0
            print("n_stages:", n_stages)
            for nth_query in range(n_stages):
                dir_checkpoints = f"{self.dir_checkpoints}/{nth_query}_query"
                os.makedirs(f"{dir_checkpoints}", exist_ok=True)

                self.log_train, self.log_val = f"{dir_checkpoints}/log_train.txt", f"{dir_checkpoints}/log_val.txt"
                write_log(f"{self.log_train}", header=["epoch", "mIoU", "pixel_acc", "loss"])
                write_log(f"{self.log_val}", header=["epoch", "mIoU", "pixel_acc"])
                self.tb_writer = SummaryWriter(log_dir=os.path.join(dir_checkpoints, 'logs'))

                self.nth_query = nth_query

                model = self._train()

                # select queries using the current model and label them.
                queries = self.query_selector(nth_query, model)
                self.dataloader.dataset.label_queries(queries, nth_query)

                if nth_query == n_stages - 1:
                    break

                # if nth_query == 0:
                #     torch.save({"model": model.state_dict()}, self.model_0_query)
        return

    def _train_epoch(self, epoch, model, optimizer, lr_scheduler):
        if self.n_pixels_by_us != 0:
            print(f"training an epoch {epoch} of {self.nth_query}th query ({self.dataloader.dataset.n_pixels_total} labelled pixels)")
            fp = f"{self.dir_checkpoints}/{self.nth_query}_query/{epoch}_train.png"
        else:
            fp = f"{self.dir_checkpoints}/fully_sup/{epoch}_train.png"
        log = f"{self.log_train}"

        dataloader_iter, tbar = iter(self.dataloader), tqdm(range(len(self.dataloader)))
        model.train()
        
        for _ in tbar:
            dict_data = next(dataloader_iter)
            x, y = dict_data['x'].to(self.device), dict_data['y'].to(self.device)
            # if queries
            if self.n_pixels_by_us != 0:
                mask = dict_data['queries'].to(self.device, torch.bool)
                y.flatten()[~mask.flatten()] = self.ignore_index

            # forward pass
            logits = model(x)
            
            '''loss_function = custom_losses()["continuity"](initial_loss=custom_losses()[self.loss],
                                                          stepsize_con=self.continuity_weight,)'''
            
            # CE
            loss_function = custom_losses()[self.loss](ignore_index=self.ignore_index)
            loss = loss_function(logits, y, )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prob, pred = F.softmax(logits.detach(), dim=1), logits.argmax(dim=1)
            self.running_score.update(y.cpu().numpy(), pred.cpu().numpy())
            self.running_loss.update(loss.detach().item())

            scores = self.running_score.get_scores()[0]
            miou, pixel_acc = scores['Mean IoU'], scores['Pixel Acc']

            # description
            description = f"({self.experim_name}) Epoch {epoch} | mIoU.: {miou:.3f} | pixel acc.: {pixel_acc:.3f} | " \
                          f"avg loss: {self.running_loss.avg:.3f}"
            tbar.set_description(description)

            if self.lr_scheduler_type == "Poly":
                lr_scheduler.step(epoch=epoch-1)

            if self.debug:
                break

        if self.lr_scheduler_type == "MultiStepLR":
            lr_scheduler.step(epoch=epoch - 1)

        write_log(log, list_entities=[epoch, miou, pixel_acc, self.running_loss.avg])
        self.tb_writer.add_scalar('mIoU/train', miou, epoch)
        self.tb_writer.add_scalar('Pixel_acc/train', pixel_acc, epoch)
        self.tb_writer.add_scalar('Loss/train', self.running_loss.avg, epoch)

        if self.running_loss.avg < self.best_loss_train:
            #state_dict = {"model": model.state_dict()}
            if self.n_pixels_by_us != 0:
                torch.save(model, f"{self.dir_checkpoints}/{self.nth_query}_query/best_train_model.pt")
            else:
                torch.save(model, f"{self.dir_checkpoints}/fully_sup/best_train_model.pt")
            print(f"best model has been saved"
                  f"(epoch: {epoch} | prev. loss: {self.best_loss_train:.4f} => new loss: {self.running_loss.avg:.4f}).")
            self.best_loss_train = self.running_loss.avg
            # Early stopping counter
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        self._reset_meters()

        ent, lc, ms, = [self._query(prob, uc)[0].cpu() for uc in ["entropy", "least_confidence", "margin_sampling"]]
        dict_tensors = {'input': dict_data['x'][0].cpu(),
                        'target': dict_data['y'][0].cpu(),
                        'pred': pred[0].detach().cpu(),
                        'confidence': lc,
                        'margin': -ms,  # minus sign is to draw smaller margin part brighter
                        'entropy': ent}

        self.vis(dict_tensors, fp=fp)
        return model, optimizer, lr_scheduler

    def _train(self):
        print(f"\n({self.experim_name}) training...\n")
        model = get_model(self.args).to(self.device)
        optimizer = get_optimizer(self.args, model)
        if self.lr_scheduler_type:
            lr_scheduler = get_lr_scheduler(self.args, optimizer=optimizer, iters_per_epoch=len(self.dataloader))
        else:
            lr_scheduler = None

        for e in range(1, 1 + self.n_epochs):
            model, optimizer, lr_scheduler = self._train_epoch(e, model, optimizer, lr_scheduler)
            self._val(e, model)
            if self.epochs_no_improve > self.early_stopping_patience:
                print('Early stopping!')
                self.epochs_no_improve = 0
                break

            if self.debug:
                break

        self.best_loss_val = float('inf')
        self.best_loss_train = float('inf')
        return model

    @torch.no_grad()
    def _val(self, epoch, model):
        dataloader_iter, tbar = iter(self.dataloader_val), tqdm(range(len(self.dataloader_val)))
        model.eval()
        for _ in tbar:
            dict_data = next(dataloader_iter)
            x, y = dict_data['x'].to(self.device), dict_data['y'].to(self.device)

            if self.dataset_name == "voc":
                h, w = y.shape[1:]
                pad_h = ceil(h / self.stride_total) * self.stride_total - x.shape[2]
                pad_w = ceil(w / self.stride_total) * self.stride_total - x.shape[3]
                x = F.pad(x, pad=(0, pad_w, 0, pad_h), mode='reflect')
                logits = model(x)[:, :, :h, :w]

            else:
                logits = model(x)
                
            loss_function = custom_losses()[self.loss](ignore_index=self.ignore_index)
            loss = loss_function(logits, y, )
            
            prob, pred = F.softmax(logits.detach(), dim=1), logits.argmax(dim=1)

            self.running_score.update(y.cpu().numpy(), pred.cpu().numpy())
            self.running_loss.update(loss.detach().item())
            scores = self.running_score.get_scores()[0]
            
            miou, pixel_acc = scores['Mean IoU'], scores['Pixel Acc']
            tbar.set_description(f"Validation: mIoU: {miou:.3f} | pixel acc.: {pixel_acc:.3f} | "
                                 f"avg loss: {self.running_loss.avg:.3f}")

            if self.debug:
                break

        if self.running_loss.avg < self.best_loss_val:
            #state_dict = {"model": model.state_dict()}
            if self.n_pixels_by_us != 0:
                torch.save(model, f"{self.dir_checkpoints}/{self.nth_query}_query/best_val_model.pt")
            else:
                torch.save(model, f"{self.dir_checkpoints}/fully_sup/best_val_model.pt")
            print(f"best model has been saved"
                  f"(epoch: {epoch} | prev. loss: {self.best_loss_val:.4f} => new loss: {self.running_loss.avg:.4f}).")
            self.best_loss_val = self.running_loss.avg

        write_log(self.log_val, list_entities=[epoch, miou, pixel_acc])
        self.tb_writer.add_scalar('mIoU/val', miou, epoch)
        self.tb_writer.add_scalar('Pixel_acc/val', pixel_acc, epoch)
        self.tb_writer.add_scalar('Loss/val', self.running_loss.avg, epoch)

        print(f"\n{'=' * 100}"
              f"\nExperim name: {self.experim_name}"
              f"\nEpoch {epoch} | miou: {miou:.3f} | pixel_acc.: {pixel_acc:.3f}"
              f"\n{'=' * 100}\n")

        self._reset_meters()

        ent, lc, ms,  = [self._query(prob, uc)[0].cpu() for uc in ["entropy", "least_confidence", "margin_sampling"]]
        dict_tensors = {'input': dict_data['x'][0].cpu(),
                        'target': dict_data['y'][0].cpu(),
                        'pred': pred[0].detach().cpu(),
                        'confidence': lc,
                        'margin': -ms,  # minus sign is to draw smaller margin part brighter
                        'entropy': ent}

        if self.n_pixels_by_us != 0:
            self.vis(dict_tensors, fp=f"{self.dir_checkpoints}/{self.nth_query}_query/{epoch}_val.png")
        else:
            self.vis(dict_tensors, fp=f"{self.dir_checkpoints}/fully_sup/{epoch}_val.png")
        return

    @staticmethod
    def _query(prob, query_strategy):
        # prob: b x n_classes x h x w
        if query_strategy == "least_confidence":
            query = 1.0 - prob.max(dim=1)[0]  # b x h x w

        elif query_strategy == "margin_sampling":
            query = prob.topk(k=2, dim=1).values  # b x k x h x w
            query = (query[:, 0, :, :] - query[:, 1, :, :]).abs()  # b x h x w

        elif query_strategy == "entropy":
            query = (-prob * torch.log(prob)).sum(dim=1)  # b x h x w

        elif query_strategy == "random":
            b, _, h, w = prob.shape
            query = torch.rand((b, h, w))

        else:
            raise ValueError
        return query

    def _reset_meters(self):
        self.running_loss.reset()
        self.running_score.reset()
