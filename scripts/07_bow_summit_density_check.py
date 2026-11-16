"""Bow Summit: what the -9999 cells in forest_scaled.tif do to the result.

Needs data/osf/BowSummit_ALOS30m_final.zip and Connaught_ALOS30m_final.zip from
OSF project zxjw5 (Sykes et al. 2024). Runs in under a minute.

In the archived Bow Summit inputs the forest density layer given to Flow-Py is
-9999 wherever basal area is 0. Flow-Py's detrainment term then reads

    rest        = max_added_detrainment_forest * forest = 0.0003 * -9999 = -3.0
    slope       = (rest - min_added_detrainment_forest) / (0 - 64.9)   = +0.046
    detrainment = max(1e-5, slope * z_delta + rest)

which is 1e-5 while z_delta is under 64.9 m and grows past the whole flux (1.0
at release) above it: 1.6 at z_delta = 100 m, 9.5 at the 270 m cap. So in these
cells a flow stops soon after its energy line height passes about 65 m. The
forest friction term is skipped (`if self.forest > 0`), so only detrainment is
affected.
"""
import importlib.util
import zipfile
from pathlib import Path

import numpy as np

from autoates_bc import classifier as C

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "06_benchmark_sites.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
ND = -9999


def main():
    for site in ("Connaught", "BowSummit"):
        d = bench.load_site(site)
        zf = zipfile.ZipFile(bench.OSF / f"{site}_ALOS30m_final.zip")
        pra = bench.read_zip(zf, "ALOS30m_final/inputdata/pra_binary.tif")[0]
        inside = d["inside"]
        nd = inside & (d["density"] == ND)
        print(f"== {site}")
        print(f"  study-area cells: {inside.sum()}, of which forest_scaled == -9999: {nd.sum()} "
              f"({100 * nd.sum() / inside.sum():.1f}%)")
        if not nd.any():
            print(f"  forest_scaled is 0 where basal area is 0: "
                  f"{int(((d['density'] == 0) & (d['forest'] == 0) & inside).sum())} cells")
            continue
        vals, cnt = np.unique(d["forest"][nd], return_counts=True)
        print("  basal area in those cells:", dict(zip(vals.tolist(), cnt.tolist())))
        print(f"  release cells in those cells: {int((nd & (pra == 1)).sum())} of {int((pra == 1).sum())}")
        base = C.Params.for_forest("bav").with_(**bench.SITES[site])
        k = (d["valid"] >= 0) & inside
        ref = bench.no0(d["valid"].astype(int))[k]
        pub = np.where(d["published"] == ND, 0, d["published"])
        as_is = bench.make_runner(d)(d["dem"], d["forest"], base)
        zeros = bench.make_runner(d, clean_density=True)(d["dem"], d["forest"], base)
        acc = lambda a: 100 * (bench.no0(a)[k] == ref).mean()
        print(f"  archived ates_gen.tif vs benchmark:              {acc(pub):.1f}%")
        print(f"  rerun, forest_scaled passed as archived:         {acc(as_is):.1f}%  "
              f"(map {100 * (as_is[inside] == pub[inside]).mean():.1f}% identical to archived)")
        print(f"  rerun, -9999 replaced by 0 (no forest):          {acc(zeros):.1f}%  "
              f"(map {100 * (zeros[inside] == pub[inside]).mean():.1f}% identical to archived)")
        for name, a in (("as archived", as_is), ("-9999 as 0", zeros)):
            per = [100 * (bench.no0(a)[k][ref == c] == c).mean() for c in range(1, 5)]
            print(f"    per class 1-4, {name}: " + ", ".join(f"{p:.1f}" for p in per))


if __name__ == "__main__":
    main()
