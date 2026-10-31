"""06 — Entrenamiento + evaluación de un experimento (Entregables 2, 3, 4 y 6).

Cada ejecución es un "run" con nombre; todo queda en 02_logs/runs/<run>/:
  config.json, historial.csv, mejor.pt, pred_val.csv, pred_dorado.csv,
  metricas_por_hallazgo.csv, metricas.json  (+ línea en 02_logs/bitacora_ejecuciones.md)

Evaluación oficial (Entregable 3): AUC-ROC macro sobre los 12 hallazgos en los estudios DORADOS,
desglose por hallazgo, IC 95 % bootstrap, y precisión / recall / F1 con umbral por hallazgo
elegido en VALIDACIÓN débil maximizando F1 (justificación en D007).

Ejemplos:
  python 06_entrenar_evaluar.py --run A1_scratch      --arch scratch  --loss bce   --ruido peso
  python 06_entrenar_evaluar.py --run A2_resnet50     --arch resnet50 --loss bce   --ruido peso
  python 06_entrenar_evaluar.py --run B1_focal        --arch resnet50 --loss focal --ruido peso
  python 06_entrenar_evaluar.py --run B2_altaconf     --arch resnet50 --loss bce   --ruido alta_conf
  python 06_entrenar_evaluar.py --run B3_coronal      --arch resnet50 --slot cor_fluid
  python 06_entrenar_evaluar.py --run C1_sin_posw     --arch resnet50 --pos_weight none
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

import config as C
import p1lib as L


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arch", default="resnet50", choices=["scratch", "resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--slot", default="sag_fluid", choices=list(C.SLOTS))
    ap.add_argument("--loss", default="bce", choices=["bce", "focal"])
    ap.add_argument("--ruido", default="peso", choices=["peso", "alta_conf", "ninguna"],
                    help="manejo de etiquetas ruidosas: ponderar por confianza | solo alta confianza | nada")
    ap.add_argument("--pos_weight", default="sqrt", choices=["sqrt", "none"],
                    help="mitigación de desbalance: pos_weight = sqrt(neg/pos) por hallazgo")
    ap.add_argument("--sampler", default="none", choices=["none", "raros"],
                    help="sobremuestreo de estudios con hallazgos raros")
    ap.add_argument("--target", default="binario", choices=["binario", "suave"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=6, help="estudios por batch (x9 cortes). Si hay OOM: --bs 4")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0, help="solo para pruebas rápidas")
    return ap.parse_args()


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, Y, I = [], [], []
    for x, y, _, i in loader:
        x = L.gpu_augment(x.to(dev, non_blocking=True), train=False)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            p = torch.sigmoid(model(x).float())
        P.append(p.cpu().numpy())
        Y.append(y.numpy())
        I.append(i.numpy())
    return np.concatenate(Y), np.concatenate(P), np.concatenate(I)


def main():
    args = get_args()
    C.seed_everything()
    run_dir = C.DIR_RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log = C.Logger(f"runs/{args.run}/train")
    C.save_json(vars(args), run_dir / "config.json")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("dispositivo:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")

    # ---------------- datos ----------------
    sub = pd.read_csv(C.DIR_DATOS / "subconjunto.csv")
    meta = pd.read_csv(C.DIR_DATOS / f"cache_meta_{args.slot}.csv")
    ok = set(meta.loc[meta["ok"] == True, C.STUDY_COL])  # noqa: E712
    sub = sub[sub[C.STUDY_COL].isin(ok)].reset_index(drop=True)
    tr_df, va_df, te_df = (sub[sub.split == s] for s in ("train", "val", "test_dorado"))
    log(f"train={len(tr_df)} val={len(va_df)} dorado={len(te_df)}")

    ds_tr = L.KneeDataset(tr_df, args.slot, args.target)
    if args.ruido == "alta_conf":
        ds_tr.w = (ds_tr.w >= 0.6).astype(np.float32)
    elif args.ruido == "ninguna":
        ds_tr.w = np.ones_like(ds_tr.w)
    log(f"fracción de etiquetas usadas (w>0): {(ds_tr.w > 0).mean():.3f}  peso medio: {ds_tr.w.mean():.3f}")

    sampler = None
    if args.sampler == "raros":
        prev = ds_tr.y.mean(0)
        raros = prev < np.quantile(prev, 0.34)       # el tercio de hallazgos menos prevalentes
        sw = 1.0 + 3.0 * (ds_tr.y[:, raros] > 0.5).any(1)
        sampler = torch.utils.data.WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.bs, shuffle=sampler is None, sampler=sampler,
                                        num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(L.KneeDataset(va_df, args.slot), batch_size=args.bs,
                                        num_workers=args.workers)
    dl_te = torch.utils.data.DataLoader(L.KneeDataset(te_df, args.slot), batch_size=args.bs,
                                        num_workers=args.workers)

    pos_weight = None
    if args.pos_weight == "sqrt":
        p = ds_tr.y.mean(0).clip(1e-3, 1 - 1e-3)
        pos_weight = torch.tensor(np.clip(np.sqrt((1 - p) / p), 1.0, 10.0), dtype=torch.float32, device=dev)
        log("pos_weight:", dict(zip(C.LABELS, np.round(pos_weight.cpu().numpy(), 2))))

    # ---------------- modelo ----------------
    model = L.KneeNet(args.arch, pretrained=not args.no_pretrained).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    lr = args.lr or (3e-4 if args.arch == "scratch" else 1e-3)
    lr_bb = lr if args.arch == "scratch" else lr / 10
    opt = torch.optim.AdamW(L.param_groups(model, lr, lr_bb), weight_decay=1e-4)
    steps = args.epochs * max(len(dl_tr), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_bb, lr], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    log(f"arch={args.arch} parámetros={n_par / 1e6:.2f}M lr_head={lr} lr_backbone={lr_bb}")

    # ---------------- entrenamiento ----------------
    best, best_ep, hist, t0, bad = -1.0, -1, [], time.time(), 0
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for step, (x, y, w, _) in enumerate(dl_tr):
            x, y, w = x.to(dev, non_blocking=True), y.to(dev), w.to(dev)
            x = L.gpu_augment(x, train=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = L.loss_fn(logits.float(), y, w, args.loss, pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
            if args.max_steps and step + 1 >= args.max_steps:
                break
        Yv, Pv, _ = predict(model, dl_va, dev)
        auc_v = L.macro_auc(Yv, Pv)
        hist.append({"epoca": ep + 1, "loss_train": tot / max(n, 1), "auc_val_debil": auc_v,
                     "min": (time.time() - t0) / 60})
        log(f"ep {ep + 1:02d} loss={tot / max(n, 1):.4f} AUC_val(débil)={auc_v:.4f} t={(time.time() - t0) / 60:.1f}min")
        if auc_v > best:
            best, best_ep, bad = auc_v, ep + 1, 0
            torch.save(model.state_dict(), run_dir / "mejor.pt")
        else:
            bad += 1
            if bad >= args.patience:
                log("early stopping")
                break
    pd.DataFrame(hist).to_csv(run_dir / "historial.csv", index=False)
    t_train = (time.time() - t0) / 60

    # ---------------- evaluación ----------------
    model.load_state_dict(torch.load(run_dir / "mejor.pt", map_location=dev))
    Yv, Pv, Iv = predict(model, dl_va, dev)
    ths = L.best_thresholds(Yv, Pv)
    t1 = time.time()
    Yt, Pt, It = predict(model, dl_te, dev)
    t_inf = (time.time() - t1) / max(len(Yt), 1)

    for nombre, (Y, P, I, df) in {"val": (Yv, Pv, Iv, va_df), "dorado": (Yt, Pt, It, te_df)}.items():
        out = pd.DataFrame(P, columns=[f"p::{l}" for l in C.LABELS])
        out.insert(0, C.STUDY_COL, df.reset_index(drop=True).loc[I, C.STUDY_COL].values)
        for j, l in enumerate(C.LABELS):
            out[f"y::{l}"] = Y[:, j]
        out.to_csv(run_dir / f"pred_{nombre}.csv", index=False)

    rep = L.per_label_report(Yt, Pt, ths)
    rep05 = L.per_label_report(Yt, Pt, np.full(C.N_LABELS, 0.5))
    rep["f1@0.5"] = rep05["f1"]
    rep.to_csv(run_dir / "metricas_por_hallazgo.csv", index=False)
    lo, hi = L.bootstrap_ci(Yt, Pt)
    m = {
        "run": args.run, "arch": args.arch, "slot": args.slot, "loss": args.loss, "ruido": args.ruido,
        "pos_weight": args.pos_weight, "sampler": args.sampler, "target": args.target,
        "parametros_M": n_par / 1e6, "mejor_epoca": best_ep, "auc_val_debil": best,
        "auc_macro_dorado": L.macro_auc(Yt, Pt), "ic95_dorado": [lo, hi],
        "f1_macro_dorado": float(rep["f1"].mean()), "precision_macro_dorado": float(rep["precision"].mean()),
        "recall_macro_dorado": float(rep["recall"].mean()),
        "auc_por_hallazgo": dict(zip(rep["hallazgo"], rep["auc"].round(4))),
        "min_entrenamiento": t_train, "seg_inferencia_por_estudio": t_inf,
        "n_train": len(tr_df), "n_val": len(va_df), "n_dorado": len(te_df),
    }
    C.save_json(m, run_dir / "metricas.json")
    log(rep.round(3).to_string(index=False))
    log(f"==> {args.run}: AUC macro DORADO = {m['auc_macro_dorado']:.4f} IC95=[{lo:.3f},{hi:.3f}] "
        f"| F1 macro = {m['f1_macro_dorado']:.3f} | {t_train:.1f} min")
    C.append_bitacora(f"06 run {args.run} ({args.arch}, {args.slot}, {args.loss}, ruido={args.ruido}, "
                      f"posw={args.pos_weight}): AUC dorado={m['auc_macro_dorado']:.4f} "
                      f"[{lo:.3f},{hi:.3f}], AUC val={best:.4f}, {t_train:.1f} min")


if __name__ == "__main__":
    main()
