import json, warnings, pathlib
import numpy as np, pandas as pd, joblib, streamlit as st
warnings.filterwarnings("ignore")
A = pathlib.Path(__file__).parent / "artifacts"
st.set_page_config(page_title="IVF Dose and Drug Decision Support (research demo)", page_icon="🧬", layout="wide")

# ---------- artifacts ----------
@st.cache_resource
def load():
    meta = json.load(open(A / "run_meta.json"))
    lg = {t: joblib.load(A / f"lgbm_l1_{t}_strict.joblib") for t in ("total_dose_iu", "start_daily_iu")}
    clfs = {p.stem.replace("lgbm_clf_", ""): joblib.load(p) for p in A.glob("lgbm_clf_*.joblib")}
    mlp = json.load(open(A / "mlp_weights.json"))
    return meta, lg, clfs, mlp
meta, LG, CLF, MLP = load()
dose_cv = pd.read_csv(A / "dose_cv_results.csv"); auc = pd.read_csv(A / "drug_choice_auc.csv"); mlp_cv = pd.read_csv(A / "mlp_challenger_results.csv")
NUM, CAT = meta["features_num"], meta["features_cat"]
TG = {"total_dose_iu": "Total stimulation dose (IU)", "start_daily_iu": "Starting daily dose (IU/day)"}
def cvmae(model, tgt):
    if model == "LightGBM":
        return float(dose_cv[(dose_cv.target == tgt) & (dose_cv.feature_set == "strict_day0")].cv_mae.iloc[0])
    return float(mlp_cv[(mlp_cv.target == tgt) & (mlp_cv.feature_set == "strict_day0")].mlp_mae.iloc[0])

# ---------- inference ----------
def enc_cats(b, row):
    out = []
    for c, cats in zip(b["cat"], b["encoder"].categories_):
        v = row[c].iloc[0]; names = [x for x in cats if isinstance(x, str)]
        out.append(np.nan if (pd.isna(v) or v not in names) else float(names.index(v)))
    return np.array([out])
def pred_lgbm(tgt, row):
    b = LG[tgt]; X = np.hstack([row[b["num"]].astype(float).values, enc_cats(b, row)])
    return float(b["model"].predict(X)[0])
def pred_mlp(tgt, row):
    m = MLP["models"][tgt]; x = row[MLP["NUM"]].astype(float).values[0]
    x = np.where(np.isnan(x), np.array(m["imp_median"]), x); x = (x - np.array(m["sc_mean"])) / np.array(m["sc_scale"])
    oh = []
    for c, cats in zip(MLP["CAT"], m["oh_cats"]):
        v = row[c].iloc[0]; v = "missing" if pd.isna(v) else str(v); oh.append(np.array([1.0 if v == k else 0.0 for k in cats]))
    h = np.concatenate([x] + oh); outs = []
    for net in m["nets"]:
        a = h
        for i in range(0, len(net), 2):
            a = np.array(net[i]) @ a + np.array(net[i + 1])
            if i < len(net) - 2: a = np.maximum(a, 0)
        outs.append(float(a[0]))
    return float(np.mean(outs) * m["y_sd"] + m["y_mu"])
def pred_clf(name, row):
    b = CLF[name]; X = np.hstack([row[b["num"]].astype(float).values, enc_cats(b, row)])
    return float(b["model"].predict_proba(X)[0, 1])

BRAND = {"label_menopur": "Menopur (hMG)", "label_menotas": "Menotas (hMG)", "label_zyhmg": "Zyhmg (hMG)", "label_hmg_generic": "hMG, generic", "label_meter0_hmg": "Meter0 (hMG, unresolved name)",
 "label_recagon": "Recagon (rFSH)", "label_gonal_f": "Gonal-f (rFSH)", "label_follisurge": "Follisurge (rFSH)", "label_briogyn": "Briogyn", "label_sitrodin": "Sitrodin",
 "label_hp_fsh_generic": "HP-FSH, generic", "label_persi0l": "Persi0l (unresolved name)", "label_pergoveris": "Pergoveris (FSH+LH)", "label_luveris": "Luveris (LH)",
 "label_growth_hormone": "Growth hormone", "label_cetrotide": "Cetrotide (antagonist)", "label_lupride": "Lupride (agonist)", "label_predmet": "Predmet"}
FLAG = {"rfsh_used": "Recombinant FSH used", "hmg_used": "hMG used", "lh_used": "LH used", "gh_used": "Growth hormone used"}

# ---------- header + model picker ----------
st.title("🧬 IVF Stimulation: Dose and Drug Decision Support")
st.caption("Research demo on retrospective clinic data. Day-0 inputs only (known before stimulation starts). Not for clinical use.")
model = st.radio("Select model", ["LightGBM", "MLP"], horizontal=True,
                 format_func=lambda m: "LightGBM (champion, gradient-boosted trees)" if m == "LightGBM" else "MLP (deep-learning challenger, neural net)")

st.subheader(f"Model scores: {model}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total dose MAE (IU)", f"{cvmae(model, 'total_dose_iu'):.1f}", help="Mean absolute error, repeated 5-fold CV on the development rows, frozen splits, strict Day-0 features. Lower is better.")
c2.metric("Start daily MAE (IU/day)", f"{cvmae(model, 'start_daily_iu'):.1f}")
c3.metric("Training rows (total / start)", "2,883 / 1,698")
c4.metric("Locked test set", "Untouched", help="580 rows held out; no model in this demo has seen or been scored on them.")
if model == "LightGBM":
    t = dose_cv[dose_cv.feature_set.isin(["strict_day0", "strict_day0_semen_dfi", "oct_core (compare)", "oct_plan (compare)"])][["target", "feature_set", "n", "cv_mae", "sd_over_runs"]].copy()
    t["target"] = t.target.map(TG); t.columns = ["Target", "Feature set", "n", "CV MAE", "SD over runs"]
    st.markdown("**Dose models (CV MAE).** `strict_day0` is what this app uses. The 'compare' rows add stim protocol or drug flags (known only after the plan is made) and are shown for reference against the October results.")
    st.dataframe(t, hide_index=True, width="stretch")
    d = auc.dropna(subset=["auc"]).copy(); d["label"] = d.label.map(lambda x: BRAND.get(x, FLAG.get(x, x)))
    d = d[["level", "label", "n", "prevalence", "auc", "auc_sd"]]; d.columns = ["Level", "Drug / class", "n", "Prevalence", "AUC-ROC", "AUC SD"]
    st.markdown("**Drug-choice classifiers (AUC-ROC, 5-fold CV).** Brand rows use the 2,275 rows where a drug is named in the record.")
    st.dataframe(d.sort_values("AUC-ROC", ascending=False), hide_index=True, width="stretch")
else:
    t = mlp_cv[["target", "feature_set", "n", "mlp_mae", "sd", "lightgbm_l1_mae", "mlp_minus_lgbm"]].copy(); t["target"] = t.target.map(TG)
    t.columns = ["Target", "Feature set", "n", "MLP MAE", "SD", "LightGBM MAE", "MLP minus LightGBM"]
    st.markdown("**MLP vs LightGBM, same frozen folds** (32-32-24-10 ReLU, dropout 0.2). Trees win on every matched set, which matches the October result.")
    st.dataframe(t, hide_index=True, width="stretch")
    st.info("The MLP predicts dose only. No AUC is reported because no drug-choice network was trained. Switch to LightGBM for drug predictions.")

# ---------- form ----------
st.divider(); st.subheader("Case input")
st.caption("Each field has a number box and a slider, kept in sync. Tick 'not measured' for tests not done: both models handle blanks.")
def dual(label, key, lo, hi, default, step, fmt="%.2f", optional=False, help=None):
    st.session_state.setdefault(key, default); st.session_state.setdefault(key + "_na", False)
    def n2s(): st.session_state[key + "_s"] = st.session_state[key + "_n"]; st.session_state[key] = st.session_state[key + "_n"]
    def s2n(): st.session_state[key + "_n"] = st.session_state[key + "_s"]; st.session_state[key] = st.session_state[key + "_s"]
    st.session_state.setdefault(key + "_n", st.session_state[key]); st.session_state.setdefault(key + "_s", st.session_state[key])
    st.markdown(f"**{label}**", help=help)
    a, b = st.columns([1, 2])
    na = st.session_state[key + "_na"]
    a.number_input(label, lo, hi, step=step, format=fmt, key=key + "_n", on_change=n2s, label_visibility="collapsed", disabled=na)
    b.slider(label, lo, hi, step=step, key=key + "_s", on_change=s2n, label_visibility="collapsed", disabled=na)
    if optional: st.checkbox("not measured", key=key + "_na")
    return np.nan if (optional and st.session_state[key + "_na"]) else float(st.session_state[key])

st.markdown("### 1. Patient profile")
p1, p2 = st.columns(2)
with p1:
    age = dual("Age (years)", "age", 18.0, 55.0, 32.0, 1.0, "%.0f")
    bmi = dual("BMI (kg/m²)", "bmi", 13.0, 60.0, 25.0, 0.1, "%.1f")
with p2:
    inf = st.selectbox("Infertility type", ["primary", "secondary"])
    indication = st.selectbox("Referral indication", ["none recorded", "failed iui", "failed ivf"])
st.markdown("### 2. Ovarian reserve and hormones")
o1, o2 = st.columns(2)
with o1:
    amh = dual("AMH (ng/mL)", "amh", 0.01, 25.0, 1.5, 0.01, "%.2f")
    afc = dual("Antral follicle count (total)", "afc", 0.0, 80.0, 8.0, 1.0, "%.0f", optional=True)
    fsh = dual("Day-2 FSH (mIU/mL)", "fsh", 0.5, 60.0, 7.0, 0.1, "%.1f", optional=True)
with o2:
    lh = dual("Basal LH (mIU/mL)", "lh", 0.1, 60.0, 5.5, 0.1, "%.1f", optional=True)
    e2 = dual("Estradiol at down-regulation (pg/mL)", "e2", 1.0, 1000.0, 45.0, 1.0, "%.0f", optional=True)
st.markdown("### 3. Treatment history")
h1, h2 = st.columns(2)
with h1:
    cyc = dual("Cycle number", "cyc", 1.0, 9.0, 1.0, 1.0, "%.0f")
    prev_any = st.radio("Previous IVF?", ["no", "yes", "not recorded"], horizontal=True)
    prev_n = dual("Previous IVF count", "pn", 0.0, 3.0, 0.0, 1.0, "%.0f", optional=True)
with h2:
    pre = st.selectbox("Pretreatment before stimulation", ["none", "antagonist", "supplements", "estradiol", "e2_testosterone", "dhea", "testosterone", "ocp", "agonist_or_downreg", "stim_regimen_text", "other", "not recorded"])

row = pd.DataFrame([dict(age=age, amh=amh, bmi=bmi, afc_total=afc, fsh_d2=fsh, lh_basal=lh, e2_downreg=e2, cycle_number=cyc,
    prev_ivf_any={"no": 0.0, "yes": 1.0}.get(prev_any, np.nan), prev_ivf_count=prev_n, infertility_type=inf,
    indication=np.nan if indication == "none recorded" else indication, pretreatment_cat=np.nan if pre == "not recorded" else pre)])

st.divider()
go = st.button("Compute prediction", type="primary", width="stretch")
if go:
    st.subheader("Predicted stimulation plan")
    other = "MLP" if model == "LightGBM" else "LightGBM"
    P = (lambda m, t: pred_lgbm(t, row) if m == "LightGBM" else pred_mlp(t, row))
    rows = []
    cols = st.columns(2)
    for col, tgt in zip(cols, ("start_daily_iu", "total_dose_iu")):
        v = P(model, tgt); v = max(v, 0.0); e = cvmae(model, tgt); step = 12.5 if tgt == "start_daily_iu" else 50
        col.metric(TG[tgt] + f" ({model})", f"{round(v / step) * step:,.0f}", help="Rounded to a practical dose step.")
        col.caption(f"Typical error of this model: about ±{e:,.0f} (MAE). Plausible range {max(v - e, 0):,.0f} to {v + e:,.0f}.")
        rows.append({"Target": TG[tgt], model: round(v), other: round(max(P(other, tgt), 0))})
    st.markdown("**Both models side by side**"); st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.warning("Honest read: dose on Day 0 sits near a noise ceiling (total dose mostly reflects how many days stimulation runs, which is unknown at Day 0). Treat the number as a starting point inside the error band, not a prescription. Mid-cycle monitoring is where tighter prediction would come from.")
    if model == "LightGBM":
        st.subheader("Predicted drug choice")
        fl = pd.DataFrame([{"Drug class": FLAG[k], "Probability": pred_clf(k, row), "AUC-ROC": float(auc[auc.label == k].auc.iloc[0])} for k in FLAG])
        st.markdown("**Drug classes**"); st.dataframe(fl.style.format({"Probability": "{:.0%}", "AUC-ROC": "{:.2f}"}), hide_index=True, width="stretch")
        br = []
        for k, nm in BRAND.items():
            a = auc[auc.label == k]
            if k in CLF and len(a) and pd.notna(a.auc.iloc[0]):
                br.append({"Brand": nm, "Probability": pred_clf(k, row), "Base rate": float(a.prevalence.iloc[0]), "AUC-ROC": float(a.auc.iloc[0])})
        br = pd.DataFrame(br).sort_values("Probability", ascending=False)
        br["Lift vs base rate"] = br.Probability / br["Base rate"]
        br["Reliability"] = np.where(br["AUC-ROC"] >= 0.75, "good", np.where(br["AUC-ROC"] >= 0.68, "moderate", "weak"))
        st.markdown("**Brands** (learned from the 2,275 rows with a named drug; probabilities are independent per brand, so they do not sum to 100%)")
        st.bar_chart(br.set_index("Brand")["Probability"], horizontal=True)
        st.dataframe(br.style.format({"Probability": "{:.0%}", "Base rate": "{:.0%}", "AUC-ROC": "{:.2f}", "Lift vs base rate": "{:.1f}x"}), hide_index=True, width="stretch")
        top = br.iloc[0]
        st.success(f"Most likely brand for this profile: {top.Brand} ({top.Probability:.0%}, AUC {top['AUC-ROC']:.2f}, reliability {top.Reliability}).")
        st.caption("34% of linked records name no drug, so brand labels are partial. Two brand names (Meter0, Persi0l) are unresolved in the source data. Brand AUCs of 0.61 to 0.84 mean ranking is better than chance but far from certain.")
    else:
        st.info("Drug-brand prediction is LightGBM-only. Switch the model selector to LightGBM to see it.")
st.caption("Trained on de-identified retrospective data. Frozen train/test splits; the locked test set was never used. Strict Day-0 features only: no stimulation protocol, drug flags or doses are used as inputs.")
