#!/usr/bin/env python3
"""
inversion_diag.py — Diagnóstico operativo de inversión térmica y capa de mezcla
a partir de wrfout completos (niveles eta).

Uso:
    python inversion_diag.py -o out/ --point 39.83,-0.23,VallUixo \
        --point 39.90,-0.28,Frente /scratch/.../wrfout_d03_2026-07-*

Salidas:
    out/inversion_diag.nc     campos 2D por paso + hora de ruptura
    out/inversion_<pt>.png    time-height de theta con inversión y capa de mezcla
    out/inversion_points.json series temporales por punto (para el visor)

Qué calcula, por columna y paso:
  inv_base / inv_top   base y techo (m AGL) de la PRIMERA capa con dtheta/dz por
                       encima del umbral, buscando desde superficie. Si la base
                       cae en el nivel más bajo, la inversión es superficial.
  inv_dtheta           theta(techo) - theta(base) [K]. Es la energía que hay que
                       vencer para romperla; el espesor solo no basta.
  mix_height           altura de mezcla por parcela (Holzworth): se sube una
                       parcela con theta_2m + exceso y se corta con el perfil.
                       NO es el PBLH del modelo, que de noche con MYNN colapsa a
                       decenas de metros y no dice nada de dónde está la tapadera.
  transport_wind       viento medio (módulo) en 0..mix_height [m/s]
  vent_index           mix_height * transport_wind [m2/s]
  breakup_hour         primer instante con mix_height > umbral, en horas desde
                       el inicio de la simulación. NaN si no rompe en el periodo.

AVISO FÍSICO: WRF no sabe que hay humo. La pluma atenúa la radiación solar,
frena el calentamiento superficial y retrasa o impide la ruptura. Por tanto
breakup_hour es el escenario MÁS OPTIMISTA (ruptura más temprana y capa más
profunda de lo real). Interpretar como cota inferior del tiempo de espera.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# ─── CONSTANTES / UMBRALES ────────────────────────────────────────────────────
G       = 9.81
RD      = 287.05
CP      = 1004.5
P0      = 100000.0

GAMMA_INV   = 0.005     # K/m — umbral de dtheta/dz para considerar capa estable
Z_SEARCH    = 3000.0    # m AGL — no se busca inversión por encima de esto
PARCEL_EXC  = 0.5       # K — exceso de la parcela sobre theta_2m (Holzworth)
BREAK_MH    = 500.0     # m AGL — umbral de "ruptura" para breakup_hour
SFC_BASED_Z = 50.0      # m AGL — base por debajo de esto ⇒ inversión superficial
# ──────────────────────────────────────────────────────────────────────────────


def wrf_times(nc):
    """Devuelve las fechas del fichero como lista de str ISO."""
    raw = nc.variables["Times"][:]
    out = []
    for row in raw:
        s = b"".join([bytes(c) for c in row]).decode("ascii") if row.dtype.kind == "S" \
            else "".join([str(c) for c in row])
        out.append(s.replace("_", "T"))
    return out


def read_step(nc, it):
    """Lee y deriva los campos necesarios de un paso temporal.

    Devuelve theta (nz,ny,nx), z_agl (nz,ny,nx), wspd (nz,ny,nx),
    theta_2m (ny,nx), hgt (ny,nx).
    """
    T     = nc.variables["T"][it]                      # theta perturbada (K)
    theta = T + 300.0

    PH  = nc.variables["PH"][it]
    PHB = nc.variables["PHB"][it]
    z_w = (PH + PHB) / G                               # geopotencial → m MSL (w-levels)
    z_m = 0.5 * (z_w[:-1] + z_w[1:])                   # a niveles de masa

    hgt   = nc.variables["HGT"][it]
    z_agl = z_m - hgt[None, :, :]

    # Destaggering del viento. No hace falta rotar a coordenadas terrestres:
    # el módulo es invariante y es lo único que usamos.
    U = nc.variables["U"][it]
    V = nc.variables["V"][it]
    u = 0.5 * (U[:, :, :-1] + U[:, :, 1:])
    v = 0.5 * (V[:, :-1, :] + V[:, 1:, :])
    wspd = np.sqrt(u * u + v * v)

    T2    = nc.variables["T2"][it]
    PSFC  = nc.variables["PSFC"][it]
    th_2m = T2 * (P0 / PSFC) ** (RD / CP)

    return (np.asarray(theta), np.asarray(z_agl), np.asarray(wspd),
            np.asarray(th_2m), np.asarray(hgt))


def inversion_layer(theta, z_agl):
    """Primera capa estable desde superficie. Vectorizado sobre (ny,nx).

    Devuelve base (m AGL), top (m AGL), dtheta (K). NaN donde no hay capa
    estable por debajo de Z_SEARCH.
    """
    nz = theta.shape[0]
    dz = np.diff(z_agl, axis=0)
    dz = np.where(dz <= 0, np.nan, dz)                 # protege niveles degenerados
    lapse = np.diff(theta, axis=0) / dz                # (nz-1, ny, nx)

    # Solo capas cuya base está dentro de la ventana de búsqueda.
    in_win = z_agl[:-1] < Z_SEARCH
    stable = (lapse >= GAMMA_INV) & in_win
    stable = np.nan_to_num(stable, nan=0).astype(bool)

    has = stable.any(axis=0)
    kb  = np.argmax(stable, axis=0)                    # primer nivel estable

    # Techo: primer nivel NO estable con índice >= kb.
    kidx    = np.arange(nz - 1)[:, None, None]
    above   = kidx >= kb[None, :, :]
    unstab  = (~stable) & above
    has_top = unstab.any(axis=0)
    kt      = np.where(has_top, np.argmax(unstab, axis=0), nz - 2)

    yy, xx = np.indices(kb.shape)
    base   = z_agl[kb, yy, xx]
    top    = z_agl[kt, yy, xx]
    dth    = theta[kt, yy, xx] - theta[kb, yy, xx]

    base = np.where(has, base, np.nan)
    top  = np.where(has, top,  np.nan)
    dth  = np.where(has, dth,  np.nan)
    return base, top, dth


def mixing_height(theta, z_agl, th_2m, excess=PARCEL_EXC):
    """Altura de mezcla por método de parcela (Holzworth), con interpolación
    lineal entre el nivel que corta y el anterior."""
    nz = theta.shape[0]
    th_p = th_2m + excess

    warmer = theta >= th_p[None, :, :]
    has    = warmer.any(axis=0)
    k      = np.argmax(warmer, axis=0)

    yy, xx = np.indices(k.shape)
    z_hi   = z_agl[k, yy, xx]
    th_hi  = theta[k, yy, xx]

    klo    = np.maximum(k - 1, 0)
    z_lo   = z_agl[klo, yy, xx]
    th_lo  = theta[klo, yy, xx]

    dth = th_hi - th_lo
    frac = np.where(np.abs(dth) > 1e-6, (th_p - th_lo) / dth, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    mh   = z_lo + frac * (z_hi - z_lo)

    # k==0: la parcela ya es más fría que el primer nivel ⇒ capa muy somera.
    mh = np.where(k == 0, z_agl[0], mh)
    # Sin corte en toda la columna ⇒ mezclado hasta el tope del dominio.
    mh = np.where(has, mh, z_agl[-1])
    return mh


def transport_and_vent(wspd, z_agl, mh):
    """Viento medio en 0..mh e índice de ventilación."""
    inlayer = z_agl <= mh[None, :, :]
    cnt     = inlayer.sum(axis=0)
    tot     = np.where(inlayer, wspd, 0.0).sum(axis=0)
    tw      = np.where(cnt > 0, tot / np.maximum(cnt, 1), wspd[0])
    return tw, mh * tw


def nearest_ij(lats, lons, lat, lon):
    d = (lats - lat) ** 2 + (lons - lon) ** 2
    return np.unravel_index(np.argmin(d), d.shape)


def parse_point(s):
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("formato: lat,lon,nombre")
    return float(parts[0]), float(parts[1]), parts[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="wrfout (niveles completos)")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--point", action="append", type=parse_point, default=[],
                    help="lat,lon,nombre (repetible)")
    ap.add_argument("--break-mh", type=float, default=BREAK_MH,
                    help=f"umbral de ruptura en m AGL (def. {BREAK_MH})")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.files)
    if not files:
        sys.exit("Sin ficheros.")

    times, fields = [], []
    lats = lons = hgt = None
    prof = {name: {"lat": la, "lon": lo, "theta": [], "z": [],
                   "inv_base": [], "inv_top": [], "inv_dtheta": [],
                   "mix_height": [], "transport_wind": [], "vent_index": []}
            for la, lo, name in args.point}
    ij = {}

    for fp in files:
        with Dataset(fp) as nc:
            tt = wrf_times(nc)
            if lats is None:
                lats = np.asarray(nc.variables["XLAT"][0])
                lons = np.asarray(nc.variables["XLONG"][0])
                for la, lo, name in args.point:
                    ij[name] = nearest_ij(lats, lons, la, lo)
                    j, i = ij[name]
                    print(f"  punto {name}: pedido ({la},{lo}) → malla "
                          f"({lats[j,i]:.4f},{lons[j,i]:.4f})")

            for it, tstr in enumerate(tt):
                theta, z_agl, wspd, th_2m, h = read_step(nc, it)
                if hgt is None:
                    hgt = h

                base, top, dth = inversion_layer(theta, z_agl)
                mh             = mixing_height(theta, z_agl, th_2m)
                tw, vi         = transport_and_vent(wspd, z_agl, mh)

                times.append(tstr)
                fields.append(dict(inv_base=base, inv_top=top, inv_dtheta=dth,
                                   mix_height=mh, transport_wind=tw, vent_index=vi))

                for name, (j, i) in ij.items():
                    p = prof[name]
                    p["theta"].append(theta[:, j, i].tolist())
                    p["z"].append(z_agl[:, j, i].tolist())
                    p["inv_base"].append(float(base[j, i]))
                    p["inv_top"].append(float(top[j, i]))
                    p["inv_dtheta"].append(float(dth[j, i]))
                    p["mix_height"].append(float(mh[j, i]))
                    p["transport_wind"].append(float(tw[j, i]))
                    p["vent_index"].append(float(vi[j, i]))

                print(f"{tstr}  mh_med={np.nanmedian(mh):7.1f} m  "
                      f"inv_top_med={np.nanmedian(top):7.1f} m  "
                      f"dth_med={np.nanmedian(dth):5.2f} K")

    nt = len(times)
    ny, nx = hgt.shape
    stack = {k: np.stack([f[k] for f in fields]) for k in fields[0]}

    # Hora de ruptura: primer paso con mix_height > umbral.
    mh_all = stack["mix_height"]
    broke  = mh_all > args.break_mh
    has_br = broke.any(axis=0)
    kbr    = np.argmax(broke, axis=0)
    dt_h   = _step_hours(times)
    breakup = np.where(has_br, kbr * dt_h, np.nan).astype(np.float32)

    ncout = out / "inversion_diag.nc"
    with Dataset(ncout, "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        ds.createDimension("strlen", 19)

        v = ds.createVariable("Times", "S1", ("time", "strlen"))
        v[:] = np.array([list(t[:19]) for t in times], dtype="S1")

        for name, arr in (("XLAT", lats), ("XLONG", lons), ("HGT", hgt)):
            v = ds.createVariable(name, "f4", ("y", "x"), zlib=True)
            v[:] = arr

        units = {"inv_base": "m AGL", "inv_top": "m AGL", "inv_dtheta": "K",
                 "mix_height": "m AGL", "transport_wind": "m s-1",
                 "vent_index": "m2 s-1"}
        for k, arr in stack.items():
            v = ds.createVariable(k, "f4", ("time", "y", "x"), zlib=True,
                                  fill_value=np.float32(np.nan))
            v[:] = arr.astype(np.float32)
            v.units = units[k]

        v = ds.createVariable("breakup_hour", "f4", ("y", "x"), zlib=True,
                              fill_value=np.float32(np.nan))
        v[:] = breakup
        v.units = "h desde inicio de simulacion"
        v.threshold_m = args.break_mh

        ds.gamma_inv   = GAMMA_INV
        ds.parcel_exc  = PARCEL_EXC
        ds.z_search    = Z_SEARCH
        ds.aviso = ("WRF sin humo: la atenuacion radiativa de la pluma retrasa la "
                    "ruptura real. breakup_hour es el escenario mas optimista.")
    print(f"\nEscrito {ncout}")

    if prof:
        pj = out / "inversion_points.json"
        pj.write_text(json.dumps({"times": times, "points": prof}))
        print(f"Escrito {pj}")

    if prof and not args.no_plots:
        _plot_points(prof, times, out)


def _step_hours(times):
    """Paso temporal en horas a partir de las dos primeras marcas."""
    if len(times) < 2:
        return 1.0
    from datetime import datetime
    f = "%Y-%m-%dT%H:%M:%S"
    t0 = datetime.strptime(times[0][:19], f)
    t1 = datetime.strptime(times[1][:19], f)
    return (t1 - t0).total_seconds() / 3600.0


def _plot_points(prof, times, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    from datetime import datetime

    tt = [datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S") for t in times]

    for name, p in prof.items():
        theta = np.array(p["theta"])          # (nt, nz)
        z     = np.array(p["z"])
        zmax  = 2500.0
        kmax  = int(np.argmax(z[0] > zmax)) or z.shape[1]

        Z = z[:, :kmax].T
        TH = theta[:, :kmax].T
        TT = np.tile(np.arange(len(tt)), (kmax, 1))

        fig, ax = plt.subplots(figsize=(11, 5.5))
        cf = ax.contourf(TT, Z, TH, levels=24, cmap="turbo")
        cs = ax.contour(TT, Z, TH, levels=12, colors="k", linewidths=0.4, alpha=0.5)
        ax.clabel(cs, fmt="%.0f", fontsize=6)

        x = np.arange(len(tt))
        ax.fill_between(x, p["inv_base"], p["inv_top"], color="w",
                        alpha=0.25, label="capa estable")
        ax.plot(x, p["inv_top"], "w-", lw=1.8)
        ax.plot(x, p["mix_height"], color="k", lw=2.2, label="altura de mezcla")

        ax.set_xticks(x[::2])
        ax.set_xticklabels([t.strftime("%d/%H") for t in tt[::2]], fontsize=8)
        ax.set_ylim(0, zmax)
        ax.set_ylabel("m sobre el terreno")
        ax.set_xlabel("día/hora UTC")
        ax.set_title(f"{name} — theta (K), inversión y capa de mezcla")
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(cf, ax=ax, label="theta (K)")
        fig.tight_layout()

        fp = out / f"inversion_{name}.png"
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        print(f"Escrito {fp}")


if __name__ == "__main__":
    main()
