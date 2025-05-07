import json
import logging
import math
import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "modality_0_features": model_out[0],
        "modality_1_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

# In your training loop, after computing features
def all_gather_features(tensor):
    tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    return tensors_gather


def save_accum_buffer(accum_buffer, epoch):
    import os
    
    # Create directory structure
    save_dir = os.path.join('logs', 'accum_buffer', f'epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each item in accum_buffer
    for idx, data_batches in enumerate(accum_buffer):
        batch_save_dir = os.path.join(save_dir, f'batch_{idx}')
        os.makedirs(batch_save_dir, exist_ok=True)
        for batch_i, (modality_0, modality_1) in enumerate(data_batches):
            modality_0_path = os.path.join(batch_save_dir, f'modality_{batch_i}_0.pt')
            modality_1_path = os.path.join(batch_save_dir, f'modality_{batch_i}_1.pt')
            torch.save(modality_0.cpu(), modality_0_path)
            torch.save(modality_1.cpu(), modality_1_path)
        if idx == 8:
            break

def inference_on_saved_accum_buffer(model, epoch, step, args):
    import os
    
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    # Load accum_buffer
    accum_buffer_dir = os.path.join('logs', 'accum_buffer', f'epoch_{epoch}')
    if not os.path.exists(accum_buffer_dir):
        print(f"No saved accum_buffer found at {accum_buffer_dir}")
        return
    
    batch_dirs = [d for d in os.listdir(accum_buffer_dir) if os.path.isdir(os.path.join(accum_buffer_dir, d))]
    for batch_dir in batch_dirs:
        batch_path = os.path.join(accum_buffer_dir, batch_dir)
        modality_files = os.listdir(batch_path)
        # Assuming the files are named consistently
        modality_0_files = sorted([f for f in modality_files if '_0.pt' in f])
        modality_1_files = sorted([f for f in modality_files if '_1.pt' in f])
        
        data_batches = []
        for m0_file, m1_file in zip(modality_0_files, modality_1_files):
            modality_0 = torch.load(os.path.join(batch_path, m0_file))
            modality_1 = torch.load(os.path.join(batch_path, m1_file))
            if modality_0.dim() > 2:
                modality_0 = modality_0.to(device, dtype=input_dtype)
            else:
                modality_0 = modality_0.to(device)
            if modality_1.dim() > 2:
                modality_1 = modality_1.to(device, dtype=input_dtype)
            else:
                modality_1 = modality_1.to(device)
            data_batches.append((modality_0, modality_1))
        
        # Perform inference
        with torch.no_grad():
            with autocast():
                model_out_list = model(data_batches)
                for jj, (model_out, modality_type) in enumerate(model_out_list):
                    # Save features
                    features_dir = os.path.join('logs', 'features', f'epoch_{epoch}', f'step_{step}', batch_dir)
                    os.makedirs(features_dir, exist_ok=True)
                    for key, val in model_out.items():
                        feature_path = os.path.join(features_dir, f'{modality_type}_{key}_{jj}.npy')
                        np.save(feature_path, val.detach().cpu().float().numpy())

def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    # set epoch in process safe manner via sampler or shared_epoch
    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # if args.accum_freq > 1:
    #     accum_modality_0, accum_modality_1, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    accum_buffer = []
    accum_buffer_neg = []
    class_labels_all = []
    accum_buffer_new = []

    if epoch == 2:
        for pg in optimizer.param_groups:
            pg['lr'] = pg['lr'] * 0.3
    
    if epoch >= 2:
        use_hn = True
    else:
        use_hn = False
    #  {i: [] for i in range(len(dataloader.dataloaders))}  # Buffer to store batches by type
    # logit_scale_text = torch.tensor(0)
    # logit_scale_image = torch.tensor(0)
    # logit_scale = torch.tensor(0)

    # save_buffer = True
    for i, batch in enumerate(dataloader):
        # if i == 32*10+1:
        #     break
        # if i == 64*40+1:
        #     break
        if i == args.accum_freq * args.epochs_steps:
            break

        accum_buffer.append([])
        accum_buffer_neg.append([])
        for batch_i, data_batch in enumerate(batch):
            if len(data_batch) == 3:
                modality_0, modality_1, class_labels = data_batch
                modality_0 = modality_0.to(
                        device=device, dtype=input_dtype, non_blocking=True)
                modality_1 = modality_1.to(device=device, non_blocking=True)
                class_labels = class_labels.to(device=device, non_blocking=True)
                class_labels_all.append(class_labels)
                accum_buffer_new.append([modality_0, modality_1])
            else:
                modality_0, neg_modality_0, modality_1, neg_modality_1 = data_batch
                if modality_0.dtype == torch.float32:
                    modality_0 = modality_0.to(
                        device=device, dtype=input_dtype, non_blocking=True)
                else:
                    modality_0 = modality_0.to(device=device, non_blocking=True)
                # print('ccc')
                if modality_1.dtype == torch.float32:
                    modality_1 = modality_1.to(
                        device=device, dtype=input_dtype, non_blocking=True)
                else:
                    modality_1 = modality_1.to(device=device, non_blocking=True)
                    
                if neg_modality_0.dtype == torch.float32:
                    neg_modality_0 = neg_modality_0.to(
                        device=device, dtype=input_dtype, non_blocking=True)
                else:
                    neg_modality_0 = neg_modality_0.to(device=device, non_blocking=True)
                if neg_modality_1.dtype == torch.float32:
                    neg_modality_1 = neg_modality_1.to(
                        device=device, dtype=input_dtype, non_blocking=True)
                else:
                    neg_modality_1 = neg_modality_1.to(device=device, non_blocking=True)
                # np.save(f'logs/tmp_files/modality{batch_i}_0.npy', modality_0.detach().cpu().float().numpy())
                # np.save(f'logs/tmp_files/modality{batch_i}_1.npy', modality_1.detach().cpu().float().numpy())
                accum_buffer[-1].append([modality_0, modality_1])
                accum_buffer_neg[-1].append([neg_modality_0, neg_modality_1])
        if not (i + 1) % args.accum_freq == 0:
            continue

        sample_index = random.randint(0, 128//8-1)
        # if epoch == 0 and save_buffer:
        #     save_accum_buffer(accum_buffer, epoch)
        #     save_buffer = False
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum
        if not args.skip_scheduler:
            scheduler(step)


        # For each dataloader's batch in the combined batch
        accum_features = {}
        accum_features_neg = {}
        for data_batches in accum_buffer:
            with torch.no_grad():
                with autocast():
                    # print('a')
                    model_out_list = model(data_batches, sample_index)
                    for out_i, (model_out, modality_type) in enumerate(model_out_list):
                        for f in ("logit_scale", "logit_bias", "logit_scale_image", "logit_bias_image", "logit_scale_text", "logit_bias_text"):
                            model_out.pop(f, None)
                        for key, val in model_out.items():
                            if not key in accum_features:
                                accum_features[key] = []
                            if out_i == 0:
                                accum_features[key].append([])
                            
                            accum_features[key][-1].append(val.detach())
        for data_batches in accum_buffer_neg:
            with torch.no_grad():
                with autocast():
                    # print('a')
                    model_out_list = model(data_batches, sample_index)
                    for out_i, (model_out, modality_type) in enumerate(model_out_list):
                        for f in ("logit_scale", "logit_bias", "logit_scale_image", "logit_bias_image", "logit_scale_text", "logit_bias_text"):
                            model_out.pop(f, None)
                        for key, val in model_out.items():
                            if not key in accum_features_neg:
                                accum_features_neg[key] = []
                            if out_i == 0:
                                accum_features_neg[key].append([])
                            
                            accum_features_neg[key][-1].append(val.detach())

        accum_features_new = {}
        for data_batches in accum_buffer_new:
            with torch.no_grad():
                with autocast():
                    model_out, _ = model([data_batches], sample_index)[0]
                    for f in ("logit_scale", "logit_bias", "logit_scale_image", "logit_bias_image", "logit_scale_text", "logit_bias_text"):
                        model_out.pop(f, None)
                    for key, val in model_out.items():
                        if not key in accum_features_new:
                            accum_features_new[key] = []
                        accum_features_new[key].append(val.detach())


        flattened_features = {}
        flattened_labels = {}
        for key in accum_features_new.keys():
            flattened_features[key] = torch.cat(accum_features_new[key])  # Shape: [N, feature_dim]
            flattened_labels[key] = torch.cat(class_labels_all)  # Shape: [N]

        # torch.cuda.empty_cache()
        # Ready to perform backprop on accumulated features
        # optimizer.zero_grad()

        # if is_distributed:
        #     torch.distributed.barrier()
        #     all_accum_features = {}
        #     for key in accum_features:
        #         all_accum_features[key] = []
        #         for accum_i in range(len(accum_features[key])):
        #             all_accum_features[key].append([])
        #             for accum_j in range(len(accum_features[key][0])):
        #                 all_accum_features[key][accum_i] += all_gather_features(accum_features[key][accum_i][accum_j])
        # else:
            # all_accum_features = accum_features
        # print(args.accum_freq, len(accum_buffer))
        # print(len(all_accum_features[key]), len(all_accum_features[key][0]), len(accum_features[key][0]))
        # print(len(accum_features['image_0_features']), len(accum_features['image_0_features'][0]))
        # logging.info(unwrap_model(model).logit_bias_image)
        # for jj in range(len((accum_buffer[0]))):
        optimizer.zero_grad()
        for j in range(len(accum_buffer)):
            with autocast():
                model_out_list = model(accum_buffer[j], sample_index, True)
                all_loss = 0
                losses = {}
                for jj, (model_out, modality_type) in enumerate(model_out_list):
                    if modality_type == 'image':
                        logit_scale_image = model_out.pop("logit_scale_image")
                        logit_bias_image = model_out.pop("logit_bias_image")
                        loss_img_image_0 = model_out.pop("loss_img_image_0") * 0.2
                        # losses["dino_loss"] = loss(**model_out, output_dict=True)
                        losses["loss_img_image_0"] = loss_img_image_0
                        all_loss += loss_img_image_0
                        inputs_no_accum = {"logit_scale": logit_scale_image, "logit_bias": logit_bias_image}
                    elif modality_type == 'text':
                        logit_scale_text = model_out.pop("logit_scale_text")
                        logit_bias_text = model_out.pop("logit_bias_text")
                        inputs_no_accum = {"logit_scale": logit_scale_text, "logit_bias": logit_bias_text}
                        loss_txt_text_0 = model_out.pop("loss_txt_text_0") * 0.2
                        all_loss += loss_txt_text_0
                        losses["loss_txt_text_0"] = loss_txt_text_0
                    else:
                        logit_scale = model_out.pop("logit_scale")
                        logit_bias = model_out.pop("logit_bias")
                        inputs_no_accum = {"logit_scale": logit_scale, "logit_bias": logit_bias}
                    
                    inputs = {}
                    for key in accum_features.keys():
                        accum_0 = [item[jj] for item in accum_features[key][:j]] if j > 0 else []
                        accum_1 = [item[jj] for item in accum_features[key][j+1:]]if j < len(accum_features[key])-1 else []
                        inputs[key] = torch.cat(accum_0 + [model_out[key]] + accum_1)
                    ct_loss = loss(**inputs, **inputs_no_accum, output_dict=True)

                    if use_hn:
                        inputs = {}
                        all_keys = list(accum_features.keys())
                        key = all_keys[0]
                        accum_0 = [item[jj] for item in accum_features[key][:j]] if j > 0 else []
                        accum_1 = [item[jj] for item in accum_features[key][j+1:]]if j < len(accum_features[key])-1 else []
                        inputs[key] = torch.cat(accum_0 + [model_out[key]] + accum_1)
                        key = all_keys[1]
                        inputs[key] = torch.cat([item[jj] for item in accum_features_neg[key]])
                        ct_loss += 0.3 * loss(**inputs, **inputs_no_accum, output_dict=True, use_hn=True)

                        inputs = {}
                        key = all_keys[1]
                        accum_0 = [item[jj] for item in accum_features[key][:j]] if j > 0 else []
                        accum_1 = [item[jj] for item in accum_features[key][j+1:]]if j < len(accum_features[key])-1 else []
                        inputs[key] = torch.cat(accum_0 + [model_out[key]] + accum_1)
                        key = all_keys[0]
                        inputs[key] = torch.cat([item[jj] for item in accum_features_neg[key]])
                        ct_loss += 0.3 * loss(**inputs, **inputs_no_accum, output_dict=True, use_hn=True)


                    if modality_type == 'image' or modality_type == 'text':
                        ct_loss = ct_loss * 0.1
                    losses[f'loss_{modality_type}'] = ct_loss
                    if not torch.isnan(ct_loss).any():
                        all_loss += ct_loss
                    else:
                        logging.info('ct loss nan')
                
                all_loss = all_loss/len(model_out_list) if len(model_out_list) > 0 else all_loss
                losses['loss_avg'] = all_loss
                backward(all_loss, scaler)

            # Clear the buffer after processing each type
        if is_distributed:
            torch.distributed.barrier()
        

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()
        
        accum_buffer.clear()
        accum_buffer_neg.clear()
        accum_features.clear()
        accum_buffer_new.clear()
        class_labels_all.clear()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
            unwrap_model(model).logit_scale_image.clamp_(0, math.log(100))
            unwrap_model(model).logit_scale_text.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args):
        # and ((i_accum + 1) % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            # inference_on_saved_accum_buffer(model, epoch, i, args)
            
            batch_size = len(modality_0)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            # logit_scale_scalar = logit_scale.item()
            # logit_scale_image_scalar = logit_scale_image.item()
            # logit_scale_text_scalar = logit_scale_text.item()
            # logit_bias_scalar = logit_bias.item()
            # logit_bias_image_scalar = logit_bias_image.item()
            # logit_bias_text_scalar = logit_bias_text.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * \
                args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, "
                f"{samples_per_second:.2f}/s, {samples_per_second_per_gpu:.2f}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:.5f} "
                # f"Logit Scale: {logit_scale_scalar:.3f} "
                # f"Logit Scale Image: {logit_scale_image_scalar:.3f} "
                # f"Logit Scale Text: {logit_scale_text_scalar:.3f} "
                # f"Logit Bias: {logit_bias_scalar:.3f} "
                # f"Logit Bias Image: {logit_bias_image_scalar:.3f} "
                # f"Logit Bias Text: {logit_bias_text_scalar:.3f} "
                f"{loss_log}"
            )  

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                # "scale": logit_scale_scalar,
                # "scale_image": logit_scale_image_scalar,
                # "scale_text": logit_scale_text_scalar,
                # "bias": logit_bias_scalar,
                # "bias_image": logit_bias_image_scalar,
                # "bias_text": logit_bias_text_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
            
        torch.cuda.empty_cache()

    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(
        model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_modality_0_features, all_modality_1_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                modality_0, modality_1 = batch
                modality_0 = modality_0.to(
                    device=device, dtype=input_dtype, non_blocking=True)
                if modality_1.dim() > 2:
                    modality_1 = modality_1.to(
                        device=device, dtype=input_dtype, non_blocking=True)
                else:
                    modality_1 = modality_1.to(
                        device=device, non_blocking=True)

                with autocast():
                    model_out = model(modality_0, modality_1)
                    modality_0_features = model_out["image_features"]
                    modality_1_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_modality_0_features.append(modality_0_features.cpu())
                    all_modality_1_features.append(modality_1_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_modality_0 = logit_scale * \
                        modality_0_features @ modality_1_features.t()
                    logits_per_modality_1 = logits_per_modality_0.t()

                    batch_size = modality_0.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_modality_0, labels) +
                        F.cross_entropy(logits_per_modality_1, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_modality_0_features),
                text_features=torch.cat(all_modality_1_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch,
                 "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            num_batches_per_epoch = 0
            dataloader = data['train'].dataloader
            num_batches_per_epoch += dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @
                        text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image,
              "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
