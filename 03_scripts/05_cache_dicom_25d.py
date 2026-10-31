"""05 — DICOM -> tensores 2.5D en caché (uint8, N_SLICES x IMG_SIZE x IMG_SIZE) por estudio y plano.

Pasos por estudio (ver D005):
  1. Elegir la serie del slot (sag_fluid = sagital PD/T2 FS; cor_fluid = coronal FS) usando las
     columnas curadas de train_series.csv (Anatomical_Plane, Fluid_Sensitive). Si hay varias,
     la de más cortes.
  2. Leer todos los cortes, ordenarlos por posición física (ImagePositionPatient).
  3. Lateralidad: tag DICOM Laterality si existe; si no, signo de la coordenada x del centro de
     la imagen (DICOM LPS: +x = izquierda del paciente). Se canonicaliza para que "medial" quede
     siempre del mismo lado:
        sagital  -> orden de cortes medial -> lateral
        coronal  -> espejo horizontal en rodillas derechas (medial siempre a la izquierda)
     Por eso NUNCA se usa flip horizontal como aumento de datos (invertiría medial/lateral).
  4. Recorte físico central de CROP_MM (si PixelSpacing existe) y redimensión a IMG_SIZE.
  5. Normalización por serie: percentiles 0.5-99.5 -> [0, 255].
  6. Se guardan N_SLICES cortes muestreados uniformemente en el 80 % central de la serie.

Uso:  python 05_cache_dicom_25d.py --slot sag_fluid [--workers 4]
Salida: cache/<slot>/<StudyInstanceUID>.npy  +  05_datos/cache_meta_<slot>.csv
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

import config as C

CROP_MM = 150.0


def _series_dir(root: Path, study: str, series: str) -> Path:
    for top in ("train_series", "train_images"):
        d = root / top / str(study) / str(series)
        if d.is_dir():
            return d
    return root / "train_series" / str(study) / str(series)


def _center_x(ds) -> float:
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    ps = np.array(ds.PixelSpacing, dtype=float)
    return float(ipp[0] + 0.5 * ds.Columns * ps[1] * iop[0] + 0.5 * ds.Rows * ps[0] * iop[3])


def _to_square(img: np.ndarray, ps) -> np.ndarray:
    h, w = img.shape
    if ps is not None:
        side_r = int(round(min(CROP_MM / float(ps[0]), h)))
        side_c = int(round(min(CROP_MM / float(ps[1]), w)))
        side = min(side_r, side_c)
    else:
        side = min(h, w)
    r0, c0 = (h - side) // 2, (w - side) // 2
    img = img[r0:r0 + side, c0:c0 + side]
    return np.asarray(Image.fromarray(img.astype(np.float32)).resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR))


def procesar(args):
    root, study, series_list, slot, out_dir = args
    out = Path(out_dir) / f"{study}.npy"
    meta = {C.STUDY_COL: study, "slot": slot, "ok": False}
    try:
        # serie con más archivos
        best, best_files = None, []
        for s in series_list:
            d = _series_dir(Path(root), study, s)
            files = sorted(d.glob("*.dcm")) if d.is_dir() else []
            if len(files) > len(best_files):
                best, best_files = s, files
        if not best_files:
            meta["error"] = "sin_archivos"
            return meta
        dss = [pydicom.dcmread(str(f), force=True) for f in best_files]
        dss = [d for d in dss if hasattr(d, "PixelData")]
        d0 = dss[0]
        iop = np.array(getattr(d0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=float)
        normal = np.cross(iop[:3], iop[3:])

        def pos(d):
            ipp = getattr(d, "ImagePositionPatient", None)
            return float(np.dot(np.array(ipp, dtype=float), normal)) if ipp is not None else float(
                getattr(d, "InstanceNumber", 0))

        dss.sort(key=pos)

        # --- lateralidad ---
        lat = str(getattr(d0, "Laterality", "") or getattr(d0, "ImageLaterality", "") or "").strip().upper()
        fuente = "tag"
        if lat not in ("L", "R"):
            try:
                cx = np.mean([_center_x(d) for d in dss])
                lat, fuente = ("L" if cx > 0 else "R"), "geometria"
            except Exception:  # noqa: BLE001
                lat, fuente = "L", "desconocida"
        medial_dir = np.array([-1.0, 0, 0]) if lat == "L" else np.array([1.0, 0, 0])

        # sagital: los cortes avanzan en la dirección +normal (~eje x). Queremos el orden
        # medial -> lateral; si la normal apunta HACIA medial, el orden ascendente es
        # lateral -> medial y se invierte.
        if abs(normal[0]) > 0.5 and np.dot(normal, medial_dir) > 0:
            dss = dss[::-1]
        flip = bool(abs(iop[0]) > 0.5 and np.dot(iop[:3], medial_dir) > 0)  # medial a la derecha -> espejo

        n = len(dss)
        idx = np.round(np.linspace(0.1 * (n - 1), 0.9 * (n - 1), C.N_SLICES)).astype(int)
        ps = getattr(d0, "PixelSpacing", None)
        imgs = []
        for i in idx:
            d = dss[i]
            a = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1) or 1) + \
                float(getattr(d, "RescaleIntercept", 0) or 0)
            if a.ndim == 3:
                a = a[..., 0]
            if flip:
                a = a[:, ::-1]
            imgs.append(_to_square(a, ps))
        vol = np.stack(imgs)
        lo, hi = np.percentile(vol, [0.5, 99.5])
        vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
        np.save(out, (vol * 255).astype(np.uint8))
        meta.update(ok=True, series=best, n_cortes=n, lateralidad=lat, fuente_lat=fuente, espejo=flip,
                    fabricante=str(getattr(d0, "Manufacturer", "")), pixel_mm=float(ps[0]) if ps else np.nan)
    except Exception as e:  # noqa: BLE001
        meta["error"] = repr(e)[:200]
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="sag_fluid", choices=list(C.SLOTS))
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    log = C.Logger(f"05_cache_{args.slot}")

    root = C.data_root()
    ts = pd.read_csv(root / "train_series.csv")
    sub = pd.read_csv(C.DIR_DATOS / "subconjunto.csv")
    plane, fluid = C.SLOTS[args.slot]
    ts = ts[(ts[C.PLANE_COL] == plane) & (ts[C.FLUID_COL] == fluid)]
    series_by_study = ts.groupby(C.STUDY_COL)[C.SERIES_COL].apply(list).to_dict()
    out_dir = C.DIR_CACHE / args.slot
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(root), s, series_by_study.get(s, []), args.slot, str(out_dir)) for s in sub[C.STUDY_COL]]
    log(f"slot={args.slot} estudios={len(jobs)} workers={args.workers} -> {out_dir}")

    metas = []
    with Pool(args.workers) as pool:
        for i, m in enumerate(pool.imap_unordered(procesar, jobs, chunksize=4)):
            metas.append(m)
            if (i + 1) % 50 == 0:
                log(f"  {i + 1}/{len(jobs)}")
    meta = pd.DataFrame(metas)
    meta.to_csv(C.DIR_DATOS / f"cache_meta_{args.slot}.csv", index=False)
    ok = meta["ok"].mean()
    log(f"OK: {meta['ok'].sum()}/{len(meta)} ({ok:.1%})")
    if "fuente_lat" in meta:
        log("fuente de lateralidad:", meta["fuente_lat"].value_counts().to_dict())
    if (~meta["ok"]).any():
        log("errores (muestra):", meta.loc[~meta["ok"], "error"].head(5).tolist())
    C.append_bitacora(f"05 cache {args.slot}: {meta['ok'].sum()}/{len(meta)} estudios OK")


if __name__ == "__main__":
    main()
