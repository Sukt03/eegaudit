"""tables/tab_core_main.tex: with EEG against without EEG, with the chance level and a baseline on the same table (one column wide).
One row per headline decoder. Every number is read through results_io (gated by CITABLE). Where the paper already has a macro for a
value (registered_values, card_numbers, v11_numbers, untrained_cells, new_numbers_v2) the table prints that macro, after checking that
the macro equals the gated source at its printed precision. A value with no macro gets a local \\Cm* macro defined at the top of the
table file (\\providecommand), so nothing is typed by hand. A source that the gate holds back prints "held back".
Also writes preview/tabtest/body.tex (the bare tabular) for the size measurement.
Run: srun -n1 -t 5 python -I -u make_tab_core_main.py"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import rio, TABLES, PREV, TEXT_BLANK

get = rio.get
S = rio.load("standard/summary.json")
PFD = rio.load("ext/ueyc_v2/profile.json")
EA = get(rio.load("ext/eu_bac_v2/full_final.json"), "A_ubp_budget")
NI = rio.load_file("baselines/ubp_noise_input", "summary.json")
T5 = rio.load("baselines/text_card_v2/text_yield_card.json")
DD = rio.load("ext/d2_ds007647_v2/summary.json")
M3 = rio.load("ext/mi_card_v3/primary_lpgo/summary_mi.json")
REVE = rio.load_file("ext/mi_fm_eogreg", "reve/summary.json")
UIMG = rio.load_file("gpu_fill_20261123/image_untrained", "summary.json")
UPE = rio.load_file("gpu_fill_20261123/pe_untrained", "summary.json")
UMI = rio.load_file("gpu_fill_20261123/mi_untrained", "summary.json")
UMIREG = rio.load_file("gpu_fill_20261123/na_mi_untrained_eogreg", "summary.json")
for nm, o in dict(S=S, PFD=PFD, EA=EA, NI=NI, T5=T5, DD=DD, M3=M3, REVE=REVE, UIMG=UIMG, UPE=UPE, UMI=UMI).items():
    assert o is not None, f"source {nm} missing or not citable"
im = S["image"]; bf = im["open_weight_models"]["brainflora_native"]; pe = S["prediction_error"]; PF4 = PFD["profile"][4]
ykey = "baseline_yield" if "baseline_yield" in EA else "twin_yield"

# ------------------------------------------------------------------ existing macros
MAC = {}
for fn in ("registered_values", "card_numbers", "v11_numbers", "untrained_cells", "new_controls", "new_numbers_v2"):
    for line in (TABLES / f"{fn}.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line)
        if m:
            MAC[m.group(1)] = m.group(2)
LOCAL = {}     # local macros defined in the table file
CHECKS = []


def fmt(x, d):
    s = f"{x:.{d}f}"
    return ("$-$" + s[1:]) if s.startswith("-") else s


def M(name, src, wrap=None):
    """Existing macro, checked against the gated source value at the macro's own precision."""
    assert name in MAC, f"macro \\{name} not found"
    txt = MAC[name].replace("$", "").replace("{", "").replace("}", "")
    num = re.search(r"[-+]?\d+\.?\d*", txt).group(0)
    d = len(num.split(".")[1]) if "." in num else 0
    ok = abs(float(num) - src) <= 0.5 * 10 ** (-d) + 1e-9
    CHECKS.append((name, MAC[name], src, ok))
    assert ok, f"macro \\{name} = {MAC[name]} disagrees with its source {src}"
    raw_minus = MAC[name].startswith("-")          # registered_values stores a bare minus sign
    if wrap == "nb":
        return r"\nb{\%s}" % name
    return (r"$\%s$" % name) if raw_minus else (r"\%s{}" % name)


def L(name, src, d):
    """Local macro for a gated value that has no macro yet."""
    name = "Cm" + name
    LOCAL[name] = fmt(src, d)
    return r"\%s{}" % name


def CI(lo, hi):
    return r"{\scriptsize[" + lo.replace("{}", "") + ", " + hi.replace("{}", "") + "]}"


def tcell(key, what, sk="pooled"):
    e = T5["summary_over_seeds"][key]["beam/" + sk][what]
    return e["mean"], e["seed_t_ci95"][0], e["seed_t_ci95"][1]


HELD = "held back"
NA = "n/a"
rows = []


def block(text):
    rows.append(r"\multicolumn{5}{@{}l}{\textit{" + text + r"}} \\")


def R(label, chance, without, with_, base):
    rows.append(" & ".join([label, chance, without, with_, base]) + r" \\")


# ------------------------------------------------------------------ image (N = 200 candidate images; protocol constant)
N_IMG = 200
assert im["ceiling"]["mean"] < __import__("math").log2(N_IMG)
ch = rf"0 bits; top-1 1/{N_IMG}"
block("Image, THINGS-EEG2: bits per 40-repetition query")
R("UBP", ch, M("VUbpNoise", NI["summary"]["noise_input_bits"]["mean"]) + "$^{n}$",
  M("CardUbpRawForty", EA["raw"]["40"]["mean"]) + " " + CI(M("CardUbpRawFortyLo", EA["raw"]["40"]["lo"]), M("CardUbpRawFortyHi", EA["raw"]["40"]["hi"])),
  M("VUbpBaseForty", EA[ykey]["40"]["mean"]) + " " + CI(M("VUbpBaseFortyLo", EA[ykey]["40"]["lo"]), M("VUbpBaseFortyHi", EA[ykey]["40"]["hi"])))
y = im["yield_per_query"]
R("NICE-GA", ch, M("VNiceFloor", y["floor"]) + "$^{s}$",
  M("NStdImgYield", y["mean"], "nb") + r" \nbci{\NStdImgYieldLo}{\NStdImgYieldHi}",
  M("UNiceKForty", UIMG["nice_ga"]["k40"]["mean"]) + " " + CI(M("UNiceKFortyLo", UIMG["nice_ga"]["k40"]["ci95"][0]), M("UNiceKFortyHi", UIMG["nice_ga"]["k40"]["ci95"][1])))
M("NStdImgYieldLo", y["ci"][0]); M("NStdImgYieldHi", y["ci"][1])
R("BrainFLORA", ch, M("VBfNoise", bf["k80_noise"]["mean"]) + "$^{n}$",
  M("VBfForty", bf["k40"]["mean"]) + " " + CI(M("VBfFortyLo", bf["k40"]["ci"][0]), M("VBfFortyHi", bf["k40"]["ci"][1])),
  M("UBfKForty", UIMG["brainflora"]["k40"]["mean"]) + " " + CI(M("UBfKFortyLo", UIMG["brainflora"]["k40"]["ci95"][0]), M("UBfKFortyHi", UIMG["brainflora"]["k40"]["ci95"][1])))
rows.append(r"\addlinespace[1pt]")

# ------------------------------------------------------------------ text (N = pooled test sentences, read from the card)
block("Text, ZuCo: bits per record, five seeds")
for key, ab, lab in (("braintranslator", "Bt", "BrainTranslator"), ("dewave_style", "Dw", "DeWave-style")):
    n_txt = int(T5["models"][key]["seed0"]["yield"]["beam"]["pooled"]["n_candidates"])
    ym, ylo, yhi = tcell(key, "eeg_yield_bits")
    with_ = M(f"V{ab}PoolEegB", ym) + " " + CI(M(f"V{ab}PoolEegBLo", ylo), M(f"V{ab}PoolEegBHi", yhi))
    nz = tcell(key, "noise_yield_bits"); noise = M(f"V{ab}PoolNoiseB", nz[0]); noise_ci = " " + CI(M(f"V{ab}PoolNoiseBLo", nz[1]), M(f"V{ab}PoolNoiseBHi", nz[2]))
    if TEXT_BLANK == "shuffled":        # without EEG: shuffled pairs; baseline: noise-trained control
        without = L(f"{ab}Shuf", tcell(key, "eeg_floor_bits")[0], 3) + "$^{s}$"; base = noise + "$^{t}$" + noise_ci
    else:                               # without EEG: noise-trained control; baseline: random-init encoder (3 seeds)
        rz = tcell(key, "random_init_yield_bits")
        assert len(T5["summary_over_seeds"][key]["beam/pooled"]["random_init_yield_bits"]["per_seed"]) == 3   # dagger: three-seed value
        without = noise + "$^{t}$"
        base = M(f"V{ab}PoolRandB", rz[0]) + r"$^{\dagger}$ " + CI(M(f"V{ab}PoolRandBLo", rz[1]), M(f"V{ab}PoolRandBHi", rz[2]))
    R(lab, rf"0 bits; top-1 1/{n_txt}", without, with_, base)
rows.append(r"\addlinespace[1pt]")

# ------------------------------------------------------------------ prediction error (continuous target: no candidate set)
block("Prediction error: bits per feedback trial")
y = pe["yield_ds003458"]
R("EEGNet, ds003458", "0 bits; no top-1", M("NStdPEFloor", y["floor"]) + "$^{c}$",
  M("NStdPEYield", y["mean"]) + " " + CI(M("NStdPEYieldLo", y["ci"][0]), M("NStdPEYieldHi", y["ci"][1])),
  M("UPeEeg", UPE["untrained_seedmean"]["mean"]) + " " + CI(M("UPeEegLo", UPE["untrained_seedmean"]["ci95"][0]), M("UPeEegHi", UPE["untrained_seedmean"]["ci95"][1])))
y = PF4["raw_yield"]       # same source as tables/tab_profile.tex (the profile interval)
R("Raw-ERP ridge, ds003458", "0 bits; no top-1", M("VPeRawFloor", PF4["noise_input_floor"]["value"]) + "$^{n}$",
  L("PeRidge", y["value"], 3) + " " + CI(L("PeRidgeLo", y["ci"][0], 3), L("PeRidgeHi", y["ci"][1], 3)), NA)
c = DD["cells"]["all|pe_raw|L64"]
R("Ridge, ds007647, target not residualised", "0 bits; no top-1", L("PeSevenNoise", c["noise_floor_mean"], 3) + "$^{n}$",
  M("CardPeRaw", c["gain"]["mean"]) + " " + CI(M("CardPeRawLo", c["gain"]["ci"][0]), M("CardPeRawHi", c["gain"]["ci"][1])), NA)
rows.append(r"\addlinespace[1pt]")

# ------------------------------------------------------------------ motor imagery (N = 4 classes; protocol constant)
N_MI = 4
assert abs(M3["rows"]["raw"]["noise"]["acc"] - 1 / N_MI) < 0.02      # noise-input accuracy sits at 1/N
ch = rf"0 bits; top-1 1/{N_MI}"
mi = lambda k: M3["rows"][k]
block("Motor imagery, EEGMMIDB: bits per trial")
R("Raw voltage", ch, M("VMiNoiseRaw", mi("raw")["noise"]["mean"]) + "$^{n}$",
  M("VMiRaw", mi("raw")["eeg"]["mean"]) + " " + CI(M("VMiRawLo", mi("raw")["eeg"]["ci"][0]), M("VMiRawHi", mi("raw")["eeg"]["ci"][1])), NA)
t = UMI["saved_trained_eegnet"]
R("EEGNet, as recorded", ch, L("MiEegnetNoise", mi("eegnet")["noise"]["mean"], 4) + "$^{n}$",
  M("UMiTr", t["mean"]) + " " + CI(M("UMiTrLo", t["ci95"][0]), M("UMiTrHi", t["ci95"][1])),
  M("UMiEeg", UMI["untrained_seedmean"]["mean"]) + " " + CI(M("UMiEegLo", UMI["untrained_seedmean"]["ci95"][0]), M("UMiEegHi", UMI["untrained_seedmean"]["ci95"][1])))
e = mi("eegnet_eogreg")
base = HELD if UMIREG is None else L("MiEegnetRegBase", UMIREG["untrained_seedmean"]["mean"], 4)
R("EEGNet, proxies regressed", ch, L("MiEegnetRegNoise", e["noise"]["mean"], 4) + "$^{n}$",
  M("VMiEegnet", e["eeg"]["mean"]) + " " + CI(M("VMiEegnetLo", e["eeg"]["ci"][0]), M("VMiEegnetHi", e["eeg"]["ci"][1])), base)
f = REVE["rows"]["fm"]
R("REVE (disjoint), proxies regressed", ch, L("ReveNoise", f["noise"]["mean"], 4) + "$^{n}$",
  M("VFmReve", f["eeg"]["mean"]) + " " + CI(L("ReveLo", f["eeg"]["ci"][0], 3), L("ReveHi", f["eeg"]["ci"][1], 3)),
  L("ReveTwin", REVE["rows"]["twin"]["eeg"]["mean"], 3) + " "
  + CI(L("ReveTwinLo", REVE["rows"]["twin"]["eeg"]["ci"][0], 3), L("ReveTwinHi", REVE["rows"]["twin"]["eeg"]["ci"][1], 3)))

# ------------------------------------------------------------------ write
TXT_BASE = ("for text it is the noise-trained control ($^{t}$)" if TEXT_BLANK == "shuffled"
            else r"for text a randomly initialised decoder ($^{\dagger}$ three-seed value)")
TXT_WO = "" if TEXT_BLANK == "shuffled" else ", or $^{t}$ the recipe trained on noise"
CAPTION = (r"\textbf{With the EEG against without it.} Yield is the score with the EEG minus the score without it: the log score of the "
           r"decoder minus that of a target-free baseline, in bits. \emph{Chance}: a decoder that knows nothing scores "
           r"0 bits and top-1 accuracy $1/N$. \emph{Without EEG}: the blank check, the same fitted "
           r"decoder and scorer on $^{n}$ noise input, $^{s}$ shuffled pairs or $^{c}$ a circular shift" + TXT_WO + r"; With EEG minus "
           r"Without EEG can differ from the yield by at most the blank. \emph{With EEG}: "
           r"the yield (95\% interval over participants; text: over seeds). \emph{Baseline}: an "
           r"untrained encoder with a fitted read-out; " + TXT_BASE + r"; n/a: no encoder; "
           r"held back: source not yet reviewed. Blocks have different units and are not ranked.")
SETUP = r"\footnotesize\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{0.95}"
BODY = (r"\begin{tabular}{@{}lcccc@{}}\toprule" + "\n"
        r"Decoder & Chance & Without EEG & With EEG {\scriptsize[95\% CI]} & Baseline {\scriptsize[95\% CI]} \\\midrule" + "\n"
        + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")
HEAD = ("% generated by figs/src/make_tab_core_main.py; do not edit. Numbers come from gated sources (results_io.CITABLE).\n"
        "% Local macros (values with no macro elsewhere):\n"
        + "".join(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}%\n" for k, v in LOCAL.items()))
(TABLES / "tab_core_main.tex").write_text(
    HEAD + r"\begin{table*}[t]\centering" + SETUP + "\n" + r"\caption{" + CAPTION + r"}\label{tab:core}" + "\n" + BODY + r"\end{table*}" + "\n")
tt = PREV / "tabtest"; tt.mkdir(parents=True, exist_ok=True)
(tt / "body.tex").write_text(HEAD + SETUP + "\n" + BODY + "\n")
(tt / "caption.txt").write_text(CAPTION + "\n")
print(f"{len(CHECKS)} reused macros agree with their gated source; {len(LOCAL)} local macros: {LOCAL}")
print("wrote", TABLES / "tab_core_main.tex")
