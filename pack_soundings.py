#!/usr/bin/env python3
"""
pack_soundings.py — extrae perfiles verticales en un pool de puntos y los
anade al paquete del visor.

El pool sale de los focos termicos ya empaquetados (data/fires.json): se cogen
las celdas de mayor FRP imponiendo una separacion minima, para que los puntos
se repartan por el perimetro en vez de apelotonarse en el frente mas activo.
Se pueden anadir puntos fijos con --point (puesto de mando, punto de carga...),
que tienen prioridad sobre los derivados de focos.

    python3 pack_soundings.py --data data --npoints 8 wrfout_d03_*

Escribe data/soundings.json y anade la entrada "soundings" al manifest.

Por que columnas y no campos 3D
-------------------------------
pack_wrf_surface lee los campos de niveles completos porque produce mapas. Aqui
solo hacen falta unas pocas columnas, asi que se indexa directamente el punto en
netCDF (nc.variables['T'][t,:,j,i]) y se lee solo eso. Con ocho puntos el coste
es despreciable frente al del empaquetado de campos.

theta seca
----------
Con use_theta_m=1 la variable T del fichero es la theta HUMEDA menos 300, no la
seca. Se detecta por el atributo global USE_THETA_M y se corrige dividiendo por
(1 + 1.61 qv). Si el atributo no esta, se asume theta seca y se avisa: un error
del 0,6 % en theta mueve la deteccion de la inversion lo suficiente para
importar.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

RD, CP, P0 = 287.05, 1004.5, 100000.0
EPS_RV = 1.61          # Rv/Rd, para pasar de theta humeda a seca


# --------------------------------------------------------------------------
# Seleccion del pool
# --------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))


def pool_from_fires(path: str, n: int, sep_km: float, max_age_h: float):
    """Selecciona hasta n focos por FRP decreciente con separacion minima.

    Devuelve lista de dicts {name, lat, lon, frp, age_h, src}. Lista vacia si el
    fichero no existe o no tiene puntos utiles: el visor sigue funcionando sin
    sondeos, no se aborta el ciclo por esto.
    """
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except Exception as e:
        print(f"      sin focos utilizables ({e}); pool solo con puntos fijos")
        return []

    pts = [p for p in doc.get("points", [])
           if isinstance(p.get("frp"), (int, float))
           and p.get("age_h", 0) <= max_age_h]
    pts.sort(key=lambda p: -p["frp"])

    out = []
    for p in pts:
        if len(out) >= n:
            break
        if any(haversine_km(p["lat"], p["lon"], q["lat"], q["lon"]) < sep_km
               for q in out):
            continue
        out.append({"name": f"Foco {len(out)+1}", "lat": float(p["lat"]),
                    "lon": float(p["lon"]), "frp": float(p["frp"]),
                    "age_h": float(p.get("age_h", 0.0)), "src": "firms"})
    print(f"      pool desde focos: {len(out)} de {len(pts)} celdas "
          f"(separacion minima {sep_km:g} km, antiguedad <= {max_age_h:g} h)")
    return out


def parse_point(s: str):
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("formato: lat,lon,nombre")
    return {"name": parts[2].strip(), "lat": float(parts[0]),
            "lon": float(parts[1]), "frp": None, "age_h": None, "src": "fijo"}


# --------------------------------------------------------------------------
# Extraccion de columnas
# --------------------------------------------------------------------------

def nearest_ij(lats, lons, lat, lon):
    d = (lats - lat)**2 + (lons - lon)**2
    j, i = np.unravel_index(int(np.argmin(d)), d.shape)
    return int(j), int(i)


def theta_dry(nc, t, j, i, qv):
    """theta seca (K) de la columna (j,i), corrigiendo theta humeda si procede."""
    th = np.asarray(nc.variables["T"][t, :, j, i], dtype=np.float64) + 300.0
    use_m = int(getattr(nc, "USE_THETA_M", -1))
    if use_m == 1:
        th = th/(1.0 + EPS_RV*qv)
    elif use_m == -1:
        theta_dry.warned = getattr(theta_dry, "warned", False)
        if not theta_dry.warned:
            print("      AVISO: el fichero no trae USE_THETA_M; se asume theta "
                  "seca. Si el run usa use_theta_m=1, theta esta sobrestimada "
                  "en torno al 0,6 % y la inversion sale desplazada.")
            theta_dry.warned = True
    return th


def column(nc, t, j, i):
    """Perfil en (j,i): devuelve dict con z, theta, T, Td, u, v.

    z en m sobre el terreno, en niveles de masa. u y v son las componentes de la
    malla, sin rotar a coordenadas terrestres: para una direccion exacta habria
    que aplicar los factores de mapa, pero en un dominio de este tamano el error
    de rotacion es de decimas de grado y no cambia ninguna decision.
    """
    g = 9.81
    ph = np.asarray(nc.variables["PH"][t, :, j, i], dtype=np.float64)
    phb = np.asarray(nc.variables["PHB"][t, :, j, i], dtype=np.float64)
    hgt = float(nc.variables["HGT"][t, j, i])
    zf = (ph + phb)/g - hgt
    z = 0.5*(zf[:-1] + zf[1:])

    p = (np.asarray(nc.variables["P"][t, :, j, i], dtype=np.float64)
         + np.asarray(nc.variables["PB"][t, :, j, i], dtype=np.float64))
    qv = np.asarray(nc.variables["QVAPOR"][t, :, j, i], dtype=np.float64)

    th = theta_dry(nc, t, j, i, qv)
    tk = th*(p/P0)**(RD/CP)
    tc = tk - 273.15

    # punto de rocio por Bolton invertido
    e = np.maximum(np.maximum(qv, 1e-12)*p/(0.622 + 0.378*np.maximum(qv, 1e-12))
                   / 100.0, 1e-6)
    ln = np.log(e/6.112)
    td = 243.5*ln/(17.67 - ln)

    # destaggering del viento en la columna
    u = 0.5*(np.asarray(nc.variables["U"][t, :, j, i], dtype=np.float64)
             + np.asarray(nc.variables["U"][t, :, j, i+1], dtype=np.float64))
    v = 0.5*(np.asarray(nc.variables["V"][t, :, j, i], dtype=np.float64)
             + np.asarray(nc.variables["V"][t, :, j+1, i], dtype=np.float64))

    return {"z": z, "theta": th, "t": tc, "td": td, "u": u, "v": v,
            "p": p/100.0}


def parse_times(nc):
    raw = nc.variables["Times"][:]
    out = []
    for row in raw:
        s = b"".join([bytes(c) if isinstance(c, bytes) else str(c).encode()
                      for c in row]).decode("ascii", "ignore").strip()
        out.append(s.replace("_", "T") + "Z")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wrfout", nargs="+")
    ap.add_argument("--data", default="data", help="directorio data/ del visor")
    ap.add_argument("--fires", default=None,
                    help="fichero de focos (por defecto <data>/fires.json)")
    ap.add_argument("--npoints", type=int, default=8,
                    help="numero maximo de puntos derivados de focos")
    ap.add_argument("--sep-km", type=float, default=3.0,
                    help="separacion minima entre puntos del pool")
    ap.add_argument("--max-age-h", type=float, default=24.0,
                    help="antiguedad maxima de un foco para entrar al pool")
    ap.add_argument("--point", action="append", type=parse_point, default=[],
                    help="punto fijo lat,lon,nombre (repetible, tiene "
                         "prioridad sobre los focos)")
    ap.add_argument("--ztop", type=float, default=4000.0,
                    help="altura maxima del perfil guardado (m sobre el terreno)")
    args = ap.parse_args()

    fires = args.fires or os.path.join(args.data, "fires.json")

    print(f"[1/3] Seleccionando pool de puntos...")
    pool = list(args.point)
    for p in pool:
        print(f"      fijo: {p['name']} ({p['lat']:.4f}, {p['lon']:.4f})")
    room = max(0, args.npoints - len(pool))
    if room:
        for p in pool_from_fires(fires, room, args.sep_km, args.max_age_h):
            if not any(haversine_km(p["lat"], p["lon"], q["lat"], q["lon"])
                       < args.sep_km for q in pool):
                pool.append(p)
    if not pool:
        print("      pool vacio: no se escriben sondeos.")
        return 0

    from netCDF4 import Dataset

    paths = sorted(args.wrfout)
    print(f"[2/3] Extrayendo columnas de {len(paths)} fichero(s)...")

    times: list[str] = []
    series: list[list[dict]] = [[] for _ in pool]
    ij = None
    nz_keep = None

    for path in paths:
        with Dataset(path) as nc:
            if ij is None:
                lats = np.asarray(nc.variables["XLAT"][0])
                lons = np.asarray(nc.variables["XLONG"][0])
                ij = []
                for p in pool:
                    j, i = nearest_ij(lats, lons, p["lat"], p["lon"])
                    # el borde rompe el destaggering por indice i+1 / j+1
                    j = min(max(j, 0), lats.shape[0] - 2)
                    i = min(max(i, 0), lats.shape[1] - 2)
                    ij.append((j, i))
                    p["glat"] = float(lats[j, i])
                    p["glon"] = float(lons[j, i])
                    p["dist_km"] = haversine_km(p["lat"], p["lon"],
                                                p["glat"], p["glon"])
                    p["z_terrain"] = float(nc.variables["HGT"][0, j, i])
                    print(f"      {p['name']:14s} malla ({j},{i}) "
                          f"{p['glat']:.4f},{p['glon']:.4f}  "
                          f"{p['dist_km']*1000:.0f} m del pedido, "
                          f"terreno {p['z_terrain']:.0f} m")

            for it, tstr in enumerate(parse_times(nc)):
                times.append(tstr)
                for k, (j, i) in enumerate(ij):
                    col = column(nc, it, j, i)
                    if nz_keep is None:
                        nz_keep = int(np.searchsorted(col["z"], args.ztop)) + 1
                        nz_keep = min(nz_keep, col["z"].size)
                        print(f"      niveles guardados: {nz_keep} de "
                              f"{col['z'].size} (hasta {args.ztop:g} m)")
                    series[k].append({v: col[v][:nz_keep] for v in col})

    order = np.argsort(times)
    times = [times[n] for n in order]

    print("[3/3] Escribiendo soundings.json...")
    out_points = []
    for k, p in enumerate(pool):
        s = [series[k][n] for n in order]
        rec = {"name": p["name"], "src": p["src"],
               "lat": p["glat"], "lon": p["glon"],
               "req_lat": p["lat"], "req_lon": p["lon"],
               "dist_km": round(p["dist_km"], 3),
               "z_terrain": round(p["z_terrain"], 1)}
        if p["frp"] is not None:
            rec["frp"] = round(p["frp"], 1)
        for var, dec in (("z", 1), ("theta", 2), ("t", 2), ("td", 2),
                         ("u", 2), ("v", 2), ("p", 1)):
            rec[var] = [[round(float(x), dec) for x in step[var]] for step in s]
        out_points.append(rec)

    doc = {"generated_from": os.path.basename(fires),
           "nz": nz_keep, "ztop": args.ztop,
           "times": times, "points": out_points}

    sp = os.path.join(args.data, "soundings.json")
    with open(sp, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    kb = os.path.getsize(sp)/1000
    print(f"      {sp}  {len(out_points)} puntos x {len(times)} pasos x "
          f"{nz_keep} niveles  ({kb:.0f} KB)")

    mp = os.path.join(args.data, "manifest.json")
    try:
        with open(mp) as fh:
            man = json.load(fh)
        man["soundings"] = {"file": "soundings.json", "n": len(out_points),
                            "nz": nz_keep, "ztop": args.ztop}
        with open(mp, "w") as fh:
            json.dump(man, fh, indent=1)
        print(f"      manifest actualizado: {mp}")
    except Exception as e:
        print(f"      AVISO: no se pudo actualizar el manifest ({e}). "
              f"El visor no mostrara los sondeos.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
