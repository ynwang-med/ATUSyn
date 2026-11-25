import csv
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from ..config import model_args, remaining_epochs, stage_for_epoch, batch_size_for_stage
from ..data.dataset import ReMINDDataset
from ..models.backbones.ncsnpp_generator_adagn_dual_text import NCSNpp
from ..models.encoders import FrozenOpenCLIPEmbedder
from ..models.diffusion import Diffusion_Coefficients, Posterior_Coefficients, q_sample_pairs, extract
from ..models.prompts import build_prompt_brain_dynamic
from ..losses.feature import FeatureLoss
from .checkpoint import EMA, load_file, load_generator, load_stage1_best, atomic_save, rng_state, restore_rng, snapshot_cpu
from .selection import Selection, early_stopping_for
from .validation import validate


def optimizer_for(model, cfg):
    tr = cfg["train"]
    return torch.optim.Adam(model.parameters(), lr=tr["lr"], betas=tuple(tr["betas"]))


def scheduler_for(optimizer, cfg, stage):
    tr = cfg["train"]
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tr["cosine_tmax_stage"+str(stage)], eta_min=tr["eta_min"])


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def train(cfg, output, device_string="cpu", resume=None, stage1_run=None):
    if resume and stage1_run:
        raise ValueError("Choose resume OR a new Stage 2 fork")
    stagewise = cfg["train"].get("stagewise_early_stopping", False)
    save_stage1 = cfg["train"].get("save_stage1_checkpoints", True)
    if not save_stage1 and stage1_run:
        raise ValueError("Memory-only Stage 1 does not support an external Stage 1 fork")
    if stagewise and stage1_run:
        raise ValueError("Stagewise experiment starts fresh or resumes its own checkpoint; no old-run fork")
    for section, key in (("text", "clip_checkpoint"), ("feature", "medsam_checkpoint")):
        if not Path(cfg[section][key]).is_file():
            raise FileNotFoundError("Set an existing local checkpoint: " + section + "." + key)
    rank, world, local = int(os.getenv("RANK",0)), int(os.getenv("WORLD_SIZE",1)), int(os.getenv("LOCAL_RANK",0))
    device = torch.device(("cuda:"+str(local)) if world > 1 and device_string.startswith("cuda") else device_string)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    output = Path(output)
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory or explicit --resume")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    tr = cfg["train"]
    seed_all(tr["seed"]+rank, device)
    train_data = ReMINDDataset(cfg, "train")
    val_data = ReMINDDataset(cfg, "val")
    if set(train_data.case_ids) & set(val_data.case_ids):
        raise ValueError("Patient leakage")
    identity = {k: hashlib.sha256(Path(cfg["data"][k]).read_bytes()).hexdigest()
                for k in ("manifest","split")}
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, seed=tr["seed"])
    loader = None
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
    args = model_args(cfg)
    model = NCSNpp(args).to(device)
    clip = FrozenOpenCLIPEmbedder(cfg["text"]["clip_checkpoint"], device=str(device)).to(device).eval()
    ema = EMA(model, tr["ema"]["decay"]) if tr["ema"]["enabled"] else None
    source_run = Path(stage1_run).resolve() if stage1_run else None
    checkpoint = load_file(source_run/"last.pt") if source_run else (load_file(resume) if resume else None)
    if not save_stage1 and checkpoint and checkpoint.get("stage") != 2:
        raise ValueError("Memory-only Stage 1 cannot resume; only Stage 2 checkpoints are resumable")
    memory_best = None
    protocol_migration = checkpoint.get("protocol_migration") if checkpoint else None
    fork = checkpoint.get("stage2_fork") if checkpoint else None
    if source_run:
        if checkpoint.get("stage") != 1 or checkpoint.get("stage2_parent") or fork:
            raise ValueError("Fork source must be a Stage 1 run, not a Stage 2 run")
        if not 1 <= checkpoint["completed_epoch"] <= tr["stage1_epochs"]:
            raise ValueError("Invalid Stage 1 completed epoch")
        if not tr["reset_optimizer_stage2"] or not tr["reset_scheduler_stage2"]:
            raise ValueError("Explicit Stage 2 fork requires optimizer and scheduler resets")
        fork = dict(source_run=str(source_run), stage1_completed_epoch=checkpoint["completed_epoch"],
                    planned_stage1_epochs=tr["stage1_epochs"],
                    source_last_sha256=hashlib.sha256((source_run/"last.pt").read_bytes()).hexdigest())
    boundary_epoch = fork["stage1_completed_epoch"] if fork else tr["stage1_epochs"]
    stage1_completed = checkpoint.get("stage1_completed_epoch") if checkpoint else None
    stage1_stop_reason = checkpoint.get("stage1_stop_reason") if checkpoint else None
    if stagewise and stage1_completed is not None:
        if not 1 <= stage1_completed <= tr["stage1_epochs"]:
            raise ValueError("Invalid recorded Stage 1 boundary")
        boundary_epoch = stage1_completed
    final_epoch = boundary_epoch + tr["stage2_epochs"]
    completed = checkpoint["completed_epoch"] if checkpoint else 0
    stage2_parent = checkpoint.get("stage2_parent") if checkpoint else None
    if (completed > boundary_epoch or (checkpoint and checkpoint.get("stage") == 2)) and not stage2_parent:
        raise ValueError("Stage 2 resume lacks best-Stage-1 provenance; do not mix training protocols")
    if checkpoint:
        if checkpoint["config"] != cfg or checkpoint["data_identity"] != identity:
            raise ValueError("Resume config or sample/split manifest changed")
        if checkpoint["world_size"] != world:
            raise ValueError("Resume world size changed; exact rank RNG recovery is unavailable")
        load_generator(model, checkpoint)
        if ema:
            ema.shadow = {k:v.to(device) for k,v in checkpoint["ema"].items()}
    if fork and not boundary_epoch <= completed <= final_epoch:
        raise ValueError("Fork checkpoint epoch outside its recorded budget")
    stage = (1 if source_run else 2) if fork else stage_for_epoch(max(1,completed), cfg)
    if stagewise:
        if checkpoint and checkpoint.get("stage") == 2 and stage1_completed is None:
            raise ValueError("Stagewise Stage 2 resume is missing its actual Stage 1 boundary")
        if completed > final_epoch:
            raise ValueError("Checkpoint exceeds its actual phase budget")
        stage = 1 if completed <= boundary_epoch else 2
    optimizer = optimizer_for(model,cfg)
    scheduler = scheduler_for(optimizer,cfg,stage)
    selector = Selection(tr["validation"]["mode"])
    early = early_stopping_for(cfg, stage)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        selector.load_state_dict(checkpoint["selection"])
        early.load_state_dict(checkpoint["early_stopping"])
        restore_rng(checkpoint["rng"][rank],device)
        if checkpoint.get("stopped_early") and not source_run:
            raise ValueError("This run already stopped early; do not silently extend it")
    network = DistributedDataParallel(model, device_ids=[device.index] if device.type=="cuda" else None) if world>1 else model
    coeff, posterior = Diffusion_Coefficients(args,device), Posterior_Coefficients(args,device)
    feature = None
    if rank == 0:
        (output/"config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
        print("Absent split patients:",train_data.absent_cases)
        if not save_stage1:
            print("[CHECKPOINT] Initial weights remain in RAM only; checkpoint saving starts with feature-loss training. An earlier interruption requires a fresh run.", flush=True)
        if resume:
            print("[RESUME] completed_epoch={}, stage={}, next_epoch={}, selected_best_epoch={}, "
                  "early_reference_epoch={}, bad_epochs={}".format(
                      completed, stage, completed+1, selector.epoch, early.best_epoch, early.bad_epochs), flush=True)
        if stagewise:
            print("[PROTOCOL] stagewise stopping; selection={}, weights={}, EMA={}, decay={}, "
                  "stage1_warmup={}, patience={}, phase_caps={}/{}, warmup_reference={}".format(
                      tr["validation"]["monitor"], tr["validation"]["weights"], tr["ema"]["enabled"],
                      tr["ema"]["decay"], tr["early_stopping"]["stage1_warmup_epochs"],
                      tr["early_stopping"]["patience"], tr["stage1_epochs"], tr["stage2_epochs"],
                      tr["early_stopping"].get("warmup_reference", "history")), flush=True)
    epochs = range(completed+1, final_epoch+1) if fork else remaining_epochs(completed,cfg)
    for epoch in epochs:
        if stagewise and epoch > final_epoch:
            break
        new_stage = 2 if fork else stage_for_epoch(epoch,cfg)
        if stagewise:
            new_stage = 1 if epoch <= boundary_epoch else 2
        if new_stage != stage:
            if save_stage1:
                best, stage2_parent = load_stage1_best(
                    model, (source_run or output)/"best_stage1.pt", cfg, identity, selector.state_dict(), ema)
            else:
                # Use rank zero's selected snapshot on every rank, just as disk loading did.
                if world > 1:
                    selected = [memory_best]
                    dist.broadcast_object_list(selected, src=0)
                    memory_best = selected[0]
                    del selected
                best, stage2_parent = load_stage1_best(
                    model, None, cfg, identity, selector.state_dict(), ema, memory_checkpoint=memory_best)
                memory_best = None
            if source_run:
                stage2_parent["source_run"] = str(source_run)
                stage2_parent["sha256"] = hashlib.sha256((source_run/"best_stage1.pt").read_bytes()).hexdigest()
            if rank == 0:
                print("[STAGE 2] Initializing from", stage2_parent, flush=True)
                (output/"stage2_parent.json").write_text(
                    json.dumps(stage2_parent, indent=2), encoding="utf-8")
                if fork:
                    (output/"stage2_fork.json").write_text(json.dumps(fork, indent=2), encoding="utf-8")
                    print("[STAGE 2 FORK] Stage 1 completed {}; training epochs {}-{}".format(
                        boundary_epoch, boundary_epoch+1, final_epoch), flush=True)
            # Keep separate stage winners; a warm-up winner is not a Full Model.
            selector = Selection(tr["validation"]["mode"])
            if fork or stagewise:
                early = early_stopping_for(cfg, 2)
            old_state = best["scheduler"]
            if tr["reset_optimizer_stage2"]:
                optimizer = optimizer_for(model,cfg)
            else:
                optimizer.load_state_dict(best["optimizer"])
            if tr["reset_scheduler_stage2"] or tr["reset_optimizer_stage2"]:
                scheduler = scheduler_for(optimizer,cfg,2)
                if not tr["reset_scheduler_stage2"]:
                    scheduler.load_state_dict(old_state)
                    for group,lr in zip(optimizer.param_groups,old_state["_last_lr"]):
                        group["lr"] = lr
            else:
                scheduler.load_state_dict(old_state)
            del best
            stage = new_stage
        batch_size = batch_size_for_stage(cfg, stage)
        if loader is None or loader.batch_size != batch_size:
            loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler,
                                num_workers=tr["workers"], drop_last=True,
                                pin_memory=device.type == "cuda")
            if not len(loader):
                raise ValueError("No complete training batch in Stage " + str(stage))
            if rank == 0:
                print("[STAGE {}] batch_per_gpu={}, world_size={}, total_batch={}, batches_per_rank={}".format(
                    stage, batch_size, world, batch_size*world, len(loader)), flush=True)
        if stage == 2 and feature is None:
            state = rng_state(device)
            from segment_anything import sam_model_registry
            med = sam_model_registry["vit_b"](checkpoint=cfg["feature"]["medsam_checkpoint"]).to(device)
            feature = FeatureLoss(clip,med)
            restore_rng(state,device)
        sampler.set_epoch(epoch)
        network.train()
        total, count = 0.0, 0
        for batch in loader:
            source,target,label = (batch[k].to(device, non_blocking=True) for k in ("source","target","label"))
            t = torch.randint(1,args.num_timesteps,(source.size(0),),device=device)
            xt,_,eps = q_sample_pairs(coeff,target,t)
            latent = torch.randn(source.size(0),args.nz,device=device)
            prompts = build_prompt_brain_dynamic(label,max_value=cfg["text"]["max_value"],min_pixels=cfg["text"]["min_pixels"])
            if np.random.rand() < cfg["text"]["dropout"]:
                prompts = [""]*source.size(0)
            with torch.no_grad():
                text = clip(prompts)
            optimizer.zero_grad()
            x0 = network(torch.cat((xt.detach(),source,label),1),t,latent,text_emb=text).clamp(-1,1)
            eps_pred = (xt-extract(coeff.a_s_cum,t,xt.shape)*x0[:,[0]]) / extract(coeff.sigmas_cum,t,xt.shape).clamp_min(1e-4)
            loss = tr["stage"+str(stage)+"_diff_weight"]*F.mse_loss(eps_pred,eps)
            if feature is not None:
                terms = feature(x0[:,[0]],target,label)
                loss = loss + sum(cfg["feature"][k]*v for k,v in terms.items())
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            if world>1:
                dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite loss; original directional denominator has no epsilon")
            loss.backward()
            optimizer.step()
            if ema:
                ema.update(model)
            total += float(loss.detach())*source.size(0)
            count += source.size(0)
        scheduler.step()
        stats = torch.tensor([total,count],device=device,dtype=torch.float64)
        if world>1:
            dist.all_reduce(stats)
        improved,stop,metrics = False,False,None
        if epoch % tr["validation"]["every"] == 0 or epoch in (boundary_epoch, final_epoch):
            if world>1:
                dist.barrier()
            if rank == 0:
                state = rng_state(device)
                backup = None
                if tr["validation"]["weights"] == "ema":
                    backup = {k:v.detach().clone() for k,v in model.state_dict().items()}
                    model.load_state_dict(ema.shadow,strict=True)
                seed_all(tr["validation"]["seed"],device)
                metrics,rows = validate(model,clip,val_loader,coeff,posterior,args,cfg,device)
                if backup is not None:
                    model.load_state_dict(backup,strict=True)
                restore_rng(state,device)
                improved = selector.update(metrics[tr["validation"]["monitor"]],epoch)
                stage_epoch = epoch if stage == 1 else epoch-boundary_epoch
                stop = early.update(metrics[tr["early_stopping"]["monitor"]],
                                    stage_epoch if stagewise else epoch)
                with (output/("validation_%04d.csv"%epoch)).open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                with (output/"history.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(dict(epoch=epoch,stage=stage,train_loss=float(stats[0]/stats[1]),
                                            batch_size_per_gpu=batch_size,total_batch=batch_size*world,
                                            batches_per_rank=len(loader),**metrics))+"\n")
                print(epoch,stage,metrics,flush=True)
                if stagewise:
                    print("[EARLY STOP] stage={}, stage_epoch={}, monitor={}, best={}, bad_epochs={}/{}".format(
                        stage, stage_epoch, tr["early_stopping"]["monitor"], early.best,
                        early.bad_epochs, early.patience), flush=True)
            if world>1:
                result=[improved,stop,selector.state_dict(),early.state_dict()]
                dist.broadcast_object_list(result,src=0)
                improved,stop=result[:2]
                selector.load_state_dict(result[2]);early.load_state_dict(result[3])
        if stagewise and stage == 1 and (stop or epoch == boundary_epoch):
            if selector.epoch is None:
                raise ValueError("No finite Stage 1 validation winner; cannot enter Stage 2")
            stage1_completed = boundary_epoch = epoch
            stage1_stop_reason = "validation_patience" if stop else "max_epochs"
            final_epoch = epoch + tr["stage2_epochs"]
            stop = False
            if rank == 0:
                print("[STAGE 1 COMPLETE] epoch={}, reason={}, best_epoch={}; next epoch starts Stage 2".format(
                    epoch, stage1_stop_reason, selector.epoch), flush=True)
        state=rng_state(device)
        states=[None]*world
        if world>1:
            dist.all_gather_object(states,state)
        else:
            states=[state]
        if rank==0:
            payload=dict(completed_epoch=epoch,stage=stage,model=model.state_dict(),
                         ema=ema.shadow if ema else None,optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(),config=cfg,data_identity=identity,
                         world_size=world,rng=states,selection=selector.state_dict(),
                         batch_size_per_gpu=batch_size,total_batch=batch_size*world,
                         stage2_parent=stage2_parent,
                         stage2_fork=fork,
                         protocol_migration=protocol_migration,
                         stage1_completed_epoch=stage1_completed,stage1_stop_reason=stage1_stop_reason,
                         early_stopping=early.state_dict(),stopped_early=stop,
                         validation=metrics,validation_weights=tr["validation"]["weights"])
            if stage == 1 and not save_stage1:
                if improved:
                    memory_best = snapshot_cpu(payload)
            else:
                atomic_save(payload,output/"last.pt")
                if improved:
                    atomic_save(payload,output/("best_stage%d.pt"%stage))
                if epoch%tr["checkpoint_every"]==0 or epoch==boundary_epoch:
                    atomic_save(payload,output/("epoch_%04d.pt"%epoch))
        if world>1:
            dist.barrier()
        if stop:
            if rank == 0 and stagewise:
                print("[TRAINING COMPLETE] Stage 2 validation patience reached; use best_stage2.pt", flush=True)
            break
        if stagewise and epoch == final_epoch and rank == 0:
            print("[TRAINING COMPLETE] Stage 2 maximum epoch budget reached; use best_stage2.pt", flush=True)
    if world>1:
        dist.destroy_process_group()
