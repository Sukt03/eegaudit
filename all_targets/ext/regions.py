"""Shared electrode-region name lists (M3 yield map, N1 artefact yield; D1/D5 may import the same file).
Fixed by name before any outcome (EXECUTOR-2, 21 Nov 2026). Every name exists in BOTH montages:
THINGS-EEG2 (63 ch, no Fz) and ds003458 (64 ch); asserted at run time. Eight regions of 6 channels each (size-matched).
frontal_pole is ocular-sensitive (cross-reference N1 / METHOD CHANGE 5).
N1 sets (METHOD, NEW IDEAS REVIEW): ocular proxy {Fp1, Fp2, AF7, AF8}; size-matched posterior control {O1, O2, PO7, PO8} (image);
PE uses the pre-ICA EOG set of rung0_ocular.py and the posterior control {Pz, POz, O1, O2}."""
import numpy as np

REGIONS = {
    "frontal_pole": ["Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8"],
    "frontal": ["F5", "F3", "F1", "F2", "F4", "F6"],
    "central": ["FCz", "C3", "C1", "Cz", "C2", "C4"],
    "left_temporal": ["FT9", "FT7", "T7", "C5", "TP9", "TP7"],
    "right_temporal": ["FT10", "FT8", "T8", "C6", "TP10", "TP8"],
    "parietal": ["CPz", "P3", "P1", "Pz", "P2", "P4"],
    "parieto_occipital": ["P7", "P5", "PO7", "PO8", "P6", "P8"],
    "occipital": ["PO3", "POz", "PO4", "O1", "Oz", "O2"],
}
# ds004595 (60 ch) has no FT9/TP9/FT10/TP10: the temporal regions swap them for F7/CP5 and F8/CP6 (lateral neighbours not used by any
# other region); every other region is unchanged. Used only by the frozen-grid M3 PE run on ds004595.
REGIONS_DS004595 = {**REGIONS, "left_temporal": ["F7", "FT7", "T7", "C5", "CP5", "TP7"], "right_temporal": ["F8", "FT8", "T8", "C6", "CP6", "TP8"]}
N1_IMAGE = {"ocular_proxy": ["Fp1", "Fp2", "AF7", "AF8"], "posterior_control": ["O1", "O2", "PO7", "PO8"]}
N1_PE_CONTROL = ["Pz", "POz", "O1", "O2"]


def indices(ch_names, names):
    ch = list(ch_names)
    missing = [n for n in names if n not in ch]
    assert not missing, f"channels missing from montage: {missing}"
    return [ch.index(n) for n in names]


def random_regions(ch_names, size=6, n=3, seed=20261121):
    """Size-matched random reference regions (fixed seed), drawn from all channels of the montage."""
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(len(ch_names), size, replace=False).tolist()) for _ in range(n)]


# ---- D1 / D5.2 sets (EXECUTOR-3, METHOD G1 PASS B7 condition 2; appended, nothing above changed). Fixed by NAME before outcomes.
# D1 8-channel groups (10-10) are a DIFFERENT partition from the M3 6-channel REGIONS above: never compare D1 groups with M3 regions directly.
# Names absent from THINGS-EEG2 (Fz, Fpz, PO9, PO10) are dropped and reported by the caller, never substituted.
D1_GROUPS8 = {
    "occipital": ["O1", "Oz", "O2", "PO3", "POz", "PO4", "PO7", "PO8"],
    "parietal": ["P5", "P3", "P1", "Pz", "P2", "P4", "P6", "CPz"],
    "temporal": ["FT7", "T7", "TP7", "TP9", "FT8", "T8", "TP8", "TP10"],
    "central": ["FC1", "FCz", "FC2", "C3", "C1", "Cz", "C2", "C4"],
    "frontal": ["F7", "F5", "F3", "F1", "F2", "F4", "F6", "F8"],
    "frontpole": ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "Fpz"],   # ocular-sensitive (no EOG); Fpz absent in THINGS-EEG2 -> 7 ch
}
# D5.2 simulated devices. tentwenty32 = the 10-20 32-channel standard (the Alljoined card lists no montage; D5 REVIEW 2) -> label
# "10-20 32 (29 present)" on THINGS-EEG2; cyton8 contains Fp1/Fp2 (ocular-sensitive, flagged).
D5_DEVICES = {
    "tentwenty32": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5",
                    "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"],
    "emotiv14": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"],
    "cyton8": ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"],
}
D5_DEVICE_LABELS = {"tentwenty32": "10-20 32 (29 present)", "emotiv14": "Emotiv EPOC 14", "cyton8": "OpenBCI Cyton 8 (contains Fp1/Fp2: ocular-sensitive)"}


if __name__ == "__main__":
    for k, v in REGIONS.items():
        print(f"{k:18s} {', '.join(v)}")
    allc = sum(REGIONS.values(), [])
    assert len(allc) == len(set(allc)) == 48
