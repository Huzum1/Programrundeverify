import streamlit as st
from collections import Counter

# ==============================
# CONFIGURARE PAGINĂa
# ==============================
st.set_page_config(
    page_title="Verificare Loterie",
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Verificare Variante Loterie")
st.divider()

# ==============================
# FUNCȚII
# ==============================
@st.cache_data(show_spinner=False)
def parse_runde_bulk(text):
    runde = []
    for linie in text.splitlines():
        nums = [int(n) for n in linie.split(",") if n.strip().isdigit()]
        if len(nums) >= 6:  # minim 6 numere pentru o rundă
            runde.append(sorted(set(nums)))  # eliminăm duplicate și sortăm
    return runde

@st.cache_data(show_spinner=False)
def parse_variante_bulk(text):
    variante = []
    for linie in text.splitlines():
        if "," not in linie:
            continue
        try:
            idv, rest = linie.split(",", 1)
            nums = [int(n) for n in rest.split() if n.strip().isdigit()]
            if len(nums) == 3:
                variante.append({
                    "id": idv.strip(), 
                    "numere": sorted(set(nums))  # sortăm și eliminăm duplicate
                })
        except:
            continue
    return variante

# ==============================
# SESSION STATE
# ==============================
st.session_state.setdefault("runde", [])
st.session_state.setdefault("variante", [])

# ==============================
# INPUT
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Runde")
    text_runde = st.text_area(
        "Format: 1,6,7,9,44,77 (o rundă pe linie)",
        height=180,
        key="input_runde"
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Adaugă Runde", type="primary", use_container_width=True, key="add_runde"):
            st.session_state.runde += parse_runde_bulk(text_runde)
            st.rerun()
    with col_b:
        if st.button("🗑️ Șterge Runde", use_container_width=True, key="del_runde"):
            st.session_state.runde = []
            st.rerun()

with col2:
    st.header("🎲 Variante (Triplete)")
    text_variante = st.text_area(
        "Format: 1, 6 7 15",
        height=180,
        key="input_variante"
    )
    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("➕ Adaugă Variante", type="primary", use_container_width=True, key="add_var"):
            st.session_state.variante += parse_variante_bulk(text_variante)
            st.rerun()
    with col_d:
        if st.button("🗑️ Șterge Variante", use_container_width=True, key="del_var"):
            st.session_state.variante = []
            st.rerun()

# ==============================
# REZULTATE
# ==============================
st.divider()
st.header("🏆 Rezultate")

if st.session_state.runde and st.session_state.variante:
    minim = st.slider(
        "Numere minime potrivite (match):",
        min_value=2,
        max_value=3,
        value=3,
        key="slider_minim"
    )

    # === Calcul statistic ===
    total_hits = 0
    unique_hits = 0
    variant_stats = {v["id"]: 0 for v in st.session_state.variante}
    runde_acoperite = 0

    for runda in st.session_state.runde:
        rset = set(runda)
        hit_in_runda = False
        
        for v in st.session_state.variante:
            match_count = len(set(v["numere"]) & rset)
            if match_count >= minim:
                variant_stats[v["id"]] += 1
                total_hits += 1
                if not hit_in_runda:
                    unique_hits += 1
                    hit_in_runda = True
                    runde_acoperite += 1

    # ==============================
    # METRICS
    # ==============================
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    
    col_s1.metric("Runde analizate", len(st.session_state.runde))
    col_s2.metric("Variante", len(st.session_state.variante))
    col_s3.metric("Runde acoperite", f"{runde_acoperite} ({runde_acoperite/len(st.session_state.runde)*100:.1f}%)")
    col_s4.metric("Total Hit-uri", total_hits)
    col_s5.metric("Hit-uri Unice", unique_hits)

    st.divider()

    # Top Variante
    st.subheader("📊 Top 20 Variante (după număr de hit-uri)")
    sorted_variants = sorted(variant_stats.items(), key=lambda x: x[1], reverse=True)
    for vid, count in sorted_variants[:20]:
        procent = count / len(st.session_state.runde) * 100
        st.text(f"Varianta {vid:>4} → {count:>3} hit-uri ({procent:.1f}%)")

    st.divider()

    # Afisare pe runde
    st.subheader("📋 Detalii pe fiecare rundă")
    with st.container(height=400):
        for i, runda in enumerate(st.session_state.runde, 1):
            cnt = sum(1 for v in st.session_state.variante 
                     if len(set(v["numere"]) & set(runda)) >= minim)
            st.text(f"Runda {i:>3} → {cnt:>2} variante câștigătoare")

    # ==============================
    # DOWNLOAD
    # ==============================
    st.divider()
    st.subheader("⬇️ Download")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)

    with col_d1:
        st.download_button(
            "Runde", 
            "\n".join(",".join(map(str, r)) for r in st.session_state.runde),
            "runde.txt"
        )
    with col_d2:
        st.download_button(
            "Variante", 
            "\n".join(f"{v['id']}, {' '.join(map(str, v['numere']))}" for v in st.session_state.variante),
            "variante.txt"
        )
    with col_d3:
        st.download_button(
            "Toate castigurile", 
            "\n".join(f"{v['id']}, {' '.join(map(str, v['numere']))}" 
                     for v in st.session_state.variante if variant_stats[v["id"]] > 0),
            "castiguri_totale.txt"
        )
    with col_d4:
        st.download_button(
            "Top Variante", 
            "\n".join(f"{vid}, {count}" for vid, count in sorted_variants),
            "top_variante.txt"
        )

else:
    st.info("➡️ Introdu runde și variante pentru a vedea analiza.")

st.caption("Made for strategy testing • Curățat și îmbunătățit")
