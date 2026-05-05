"""
One-shot diagnostic: does Pace's `Bayes_0d8_binary.dat` (62 stars, all p ≥ 0.8)
select the same Segue 1 stars as our pipeline's `Bpr > 0.8` cut on the
Simon+2011 catalog? Cross-match by sky position, compare V/e_V on matched
stars, scatter Pace p vs Simon Bpr.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PACE = HERE / "data" / "Pace_Segue1_Bayes_0d8_binary.dat"
SIMON = HERE / "data" / "segue1_kinematics_simon2011.csv"
TOL_ARCSEC = 1.0


def main() -> None:
    P = np.loadtxt(PACE, skiprows=1)
    ra_P, dec_P, V_P, eV_P, p_P = P.T

    df = pd.read_csv(SIMON)
    df = df.dropna(subset=["Vel", "e_Vel", "Bpr", "_RA", "_DE"])
    S_all = df.reset_index(drop=True)
    S = S_all[S_all["Bpr"] > 0.8].reset_index(drop=True)

    print(f"Pace 0.8 file:        N = {len(P)}  (header claims 62)")
    print(f"Simon Bpr>0.8:        N = {len(S)}")
    print(f"Pace p min:           {p_P.min():.3f}  (header claims all >= 0.8)")
    assert (p_P >= 0.8).all(), "Pace file contains stars with p < 0.8"

    c_S = SkyCoord(S["_RA"].values, S["_DE"].values, unit=u.deg)
    c_S_all = SkyCoord(S_all["_RA"].values, S_all["_DE"].values,
                        unit=u.deg)
    c_P = SkyCoord(ra_P * u.deg, dec_P * u.deg)

    # Pace -> Simon (Bpr>0.8 set)
    idx_PS, sep_PS, _ = match_coordinates_sky(c_P, c_S)
    matched_P = sep_PS.arcsec < TOL_ARCSEC
    print()
    print(f"Pace -> Simon(Bpr>0.8) within {TOL_ARCSEC:.1f}\":")
    print(f"  matched   = {matched_P.sum()} / {len(P)}")
    print(f"  unmatched = {(~matched_P).sum()} (in Pace, not in Simon-Bpr>0.8)")

    if (~matched_P).sum():
        # For unmatched Pace stars, look in the FULL Simon catalog (no Bpr cut).
        idx_unP_all, sep_unP_all, _ = match_coordinates_sky(c_P[~matched_P], c_S_all)
        print("  unmatched Pace stars -> nearest in *full* Simon catalog:")
        for j, (i, mask_idx) in enumerate(zip(idx_unP_all, np.where(~matched_P)[0])):
            r = P[mask_idx]
            nn = S_all.iloc[int(i)]
            sep = float(sep_unP_all[j].arcsec)
            print(f"    Pace star (RA={r[0]:.4f}, Dec={r[1]:.4f}, V={r[2]:7.2f}, "
                  f"eV={r[3]:5.2f}, p={r[4]:.3f})")
            print(f"      nearest Simon (any Bpr): sep={sep:6.2f}\"  "
                  f"V={nn.Vel:7.2f}  eV={nn.e_Vel:5.2f}  Bpr={nn.Bpr:.3f}")

    # Simon -> Pace
    idx_SP, sep_SP, _ = match_coordinates_sky(c_S, c_P)
    matched_S = sep_SP.arcsec < TOL_ARCSEC
    print()
    print(f"Simon(Bpr>0.8) -> Pace within {TOL_ARCSEC:.1f}\":")
    print(f"  matched   = {matched_S.sum()} / {len(S)}")
    print(f"  unmatched = {(~matched_S).sum()} (Bpr>0.8 but absent from Pace)")

    if (~matched_S).sum():
        unS = S[~matched_S]
        idx_unS_P, sep_unS_P, _ = match_coordinates_sky(
            SkyCoord(unS["_RA"].values, unS["_DE"].values,
                     unit=u.deg),
            c_P,
        )
        print("  unmatched Simon-Bpr>0.8 stars -> nearest in Pace:")
        for j, (_, r) in enumerate(unS.iterrows()):
            i_p = int(idx_unS_P[j])
            sep = float(sep_unS_P[j].arcsec)
            print(f"    Simon star (V={r.Vel:7.2f}, eV={r.e_Vel:5.2f}, "
                  f"Bpr={r.Bpr:.3f})")
            print(f"      nearest Pace: sep={sep:6.2f}\"  V={V_P[i_p]:7.2f}  "
                  f"eV={eV_P[i_p]:5.2f}  p={p_P[i_p]:.3f}")

    # Velocity / error agreement on matched stars (Pace -> Simon-Bpr>0.8 view).
    if matched_P.sum():
        i_match_S = idx_PS[matched_P]  # Simon-Bpr>0.8 row indices, aligned to matched Pace stars
        V_S_m = S.iloc[i_match_S]["Vel"].values
        eV_S_m = S.iloc[i_match_S]["e_Vel"].values
        Bpr_S_m = S.iloc[i_match_S]["Bpr"].values
        V_P_m = V_P[matched_P]
        eV_P_m = eV_P[matched_P]
        p_P_m = p_P[matched_P]

        dV = V_P_m - V_S_m
        deV = eV_P_m - eV_S_m
        print()
        print(f"On {matched_P.sum()} matched stars:")
        print(f"  ΔV  = V_Pace − V_Simon:    median={np.median(dV):+.4f}  "
              f"max|·|={np.max(np.abs(dV)):.4f}  std={np.std(dV):.4f}  km/s")
        print(f"  Δe_V = e_Pace − e_Simon:   median={np.median(deV):+.4f}  "
              f"max|·|={np.max(np.abs(deV)):.4f}  std={np.std(deV):.4f}  km/s")
        n_disagree_V = int((np.abs(dV) > 1e-3).sum())
        n_disagree_eV = int((np.abs(deV) > 1e-3).sum())
        print(f"  |ΔV|  > 1e-3 km/s:  {n_disagree_V} stars")
        print(f"  |ΔeV| > 1e-3 km/s:  {n_disagree_eV} stars")

        rho, pval = spearmanr(p_P_m, Bpr_S_m)
        print(f"  Spearman ρ(p_Pace, Bpr_Simon) = {rho:+.4f}  (p={pval:.2e})")
        # Cells where the two cuts would disagree:
        n_pace_bpr_low = int((Bpr_S_m <= 0.8).sum())
        print(f"  matched stars with Pace p>0.8 but Simon Bpr ≤ 0.8: {n_pace_bpr_low}")
        if n_pace_bpr_low:
            mask = Bpr_S_m <= 0.8
            for v_p, v_s, bpr, pp in zip(V_P_m[mask], V_S_m[mask],
                                          Bpr_S_m[mask], p_P_m[mask]):
                print(f"    V_P={v_p:7.2f}  V_S={v_s:7.2f}  Bpr={bpr:.3f}  p_Pace={pp:.3f}")

    # Summary
    print()
    print("=== summary ===")
    if matched_P.all() and matched_S.all() and len(P) == len(S):
        print(f"Identical star sets ({len(P)}). ", end="")
    else:
        print(f"Different sets. Pace has {(~matched_P).sum()} stars not in our cut; "
              f"our cut has {(~matched_S).sum()} stars not in Pace.")
    if matched_P.sum():
        print(f"Velocities agree to median |ΔV|={np.median(np.abs(dV)):.4f} km/s "
              f"(max {np.max(np.abs(dV)):.4f}).")


if __name__ == "__main__":
    main()
