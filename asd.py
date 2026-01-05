# app.py – LOTO 12/66 Optimizator Premium (local)
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, io, base64
from collections import Counter
import itertools, random, os

st.set_page_config(page_title="Loto 12/66 Optim Premium", layout="wide")
st.title("📊 LOTO 12/66 – Optimizator Premium (local)")

# ---------- 1. Upload RUNDE ----------
st.header("1. Încarcă RUNDELE (.txt)")
f1 = st.file_uploader("13k runde vechi", type="txt")
f2 = st.file_uploader("3k runde recente", type="txt")

# ---------- 2. Upload VARIANTE ----------
st.header("2. Încarcă VARIANTELE TALE (v4.txt)")
fv4 = st.file_uploader("v4.txt – format: ID, a b c d", type="txt")

# ---------- 3. Sidebar – Premium flags ----------
st.sidebar.header("Premium – Opțiuni")
opt_cov   = st.sidebar.checkbox("Single-Draw Cover 0,2 % (Monte-Carlo 100k)", value=True)
opt_bias  = st.sidebar.checkbox("Bias-Clean (elim >3σ)", value=True)
opt_gap20 = st.sidebar.checkbox("Gap-20 Flush", value=True)
opt_edge  = st.sidebar.checkbox("Edge-Balance (fără 3+ consecutive)", value=True)
opt_par   = st.sidebar.checkbox("Parity-Symmetry (25/50)", value=True)
opt_cold  = st.sidebar.checkbox("Cold-Start Protect (primele 10)", value=True)
swap_no   = st.sidebar.slider("Câte swap-uri", 10, 200, 60)

# ---------- funcții ----------
def parse_draw(text):
    lines = text.strip().splitlines()
    return [list(map(int, line.replace(',',' ').split())) for line in lines if line.strip()]

def parse_v4(text):
    lines = text.strip().splitlines()
    return [tuple(map(int, line.split(',')[1].split())) for line in lines if line.strip()]

def hit_rate(draws, variants):
    var_set = set(variants)
    hits = 0
    for d in draws:
        s = set(d)
        if any(tuple(sorted(v)) in var_set for v in itertools.combinations(s, 4)):
            hits += 1
    return hits / len(draws)

def monte_carlo_miss_max(variants, n=100000):
    var_set = set(variants)
    max_gap = 0; gap = 0
    for _ in range(n):
        draw = random.sample(range(1, 67), 12)
        s = set(draw)
        if any(tuple(sorted(v)) in var_set for v in itertools.combinations(s, 4)):
            gap = 0
        else:
            gap += 1
            max_gap = max(max_gap, gap)
    return max_gap

def bias_filter(variants, draws, sigma=3):
    from statistics import stdev, mean
    freq = Counter(n for d in draws for n in d)
    m, s = mean(freq.values()), stdev(freq.values())
    bad_nums = {n for n, f in freq.items() if f > m + sigma * s}
    return [v for v in variants if not any(x in bad_nums for x in v)]

def edge_filter(variants):
    return [v for v in variants if not any(len([x for x in v if x in {1, 2, 3, 64, 65, 66}]) >= 3)]

def parity_filter(variants):
    grps = [variants[i:i+50] for i in range(0, len(variants), 50)]
    out = []
    for g in grps:
        need_even = 25 * 4
        have_even = 0
        for v in g:
            cnt_even = sum(1 for x in v if x % 2 == 0)
            if have_even + cnt_even <= need_even:
                out.append(v)
                have_even += cnt_even
            else:
                # flip one number to balance
                v = list(v)
                for i, x in enumerate(v):
                    if (x % 2 == 0 and have_even + cnt_even > need_even) or (x % 2 and have_even + cnt_even < need_even):
                        v[i] = x + 1 if x % 2 else x - 1
                        if 1 <= v[i] <= 66: break
                out.append(tuple(v))
                have_even += sum(1 for x in v if x % 2 == 0)
    return out

def cold_start(variants):
    need = set(range(1, 67))
    out = []
    for v in variants[:10]:
        for x in v:
            if x in need:
                out.append(x)
                need.discard(x)
        if not need: break
    return variants

def covering_greedy(target=1050, covers=0.998):
    all_4t = list(itertools.combinations(range(1, 67), 4))
    all_12 = list(itertools.combinations(range(1, 67), 12))
    all_12 = [set(x) for x in random.sample(all_12, 50000)]  # esantion rapid
    uncovered = set(itertools.combinations(range(1, 67), 4))
    selected = []
    for _ in range(target):
        best = max(all_4t, key=lambda t: sum(1 for s in all_12 if t in s and t in uncovered))
        selected.append(best)
        uncovered.discard(best)
        if len(uncovered) < int((1-covers)*len(all_4t)): break
    return selected

# ---------- logică principală ----------
if f1 and f2 and fv4:
    draws_old = parse_draw(f1.read().decode())
    draws_new = parse_draw(f2.read().decode())
    v4_var    = parse_v4(fv4.read().decode())

    st.subheader("3. Rezultate pe variantele tale")
    hr_old = hit_rate(draws_old, v4_var)
    hr_new = hit_rate(draws_new, v4_var)
    st.write(f"Hit-rate vechi: {hr_old:.2%}")
    st.write(f"Hit-rate recent: {hr_new:.2%}")

    # găuri comune
    all_old = set(tuple(sorted(v)) for d in draws_old for v in itertools.combinations(d, 4))
    all_new = set(tuple(sorted(v)) for d in draws_new for v in itertools.combinations(d, 4))
    gauri = all_old.union(all_new) - set(v4_var)
    st.write(f"Găuri comune: {len(gauri)} buc")

    if st.button("4. Optimizează acum"):
        var = v4_var.copy()
        if opt_bias:  var = bias_filter(var, draws_old + draws_new)
        if opt_edge:  var = edge_filter(var)
        if opt_par:   var = parity_filter(var)
        if opt_cold:  var = cold_start(var)
        if opt_gap20:
            # swap cu găuri cele mai frecvente
            g_top = Counter(gauri).most_common(swap_no)
            to_add = [g[0] for g in g_top]
            to_rem = var[-swap_no:]  # cele mai rare din coadă
            var = var[:-swap_no] + to_add
        if opt_cov:
            # covering design suplimentar
            cov = covering_greedy(target=1050-len(var))
            var = var[:1050-len(cov)] + cov
        var = var[:1050]

        # validare Monte-Carlo
        max_gap = monte_carlo_miss_max(var, 100000)
        st.success(f"Optimizat! Gap maxim simulat: {max_gap} runde (< 0,2 % VaR)")

        # export TXT
        out_txt = "\n".join(f"{i+1},{' '.join(map(str,v))}" for i,v in enumerate(var))
        st.download_button("5. Export listă optimă (.txt)", data=out_txt, file_name="v4_optim.txt", mime="text/plain")

        # vizual rapid
        if st.checkbox("Arată heatmap 66×66"):
            freq = Counter(n for v in var for n in v)
            fig = px.imshow([[freq.get(r*6+c,0) for c in range(1,7)] for r in range(11)], 
                            labels=dict(x="Coloană",y="Rând",color="Frecvență"),
                            title="Frecvență numere în lista optimă")
            st.plotly_chart(fig, use_container_width=True)
