import math
import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy.utils import iers

# Avoid network dependency for Earth-orientation updates during normal use.
iers.conf.auto_download = False

st.set_page_config(page_title="Astro Planner", page_icon="🌌", layout="wide")

SITES = {
    "Starfront — Texas": {
        "lat": 31.546944,
        "lon": -99.381944,
        "elev": 500,
        "tz": "America/Chicago",
    },
    "Hakos Guest Farm — Namibie": {
        "lat": -23.233333,
        "lon": 16.366667,
        "elev": 1800,
        "tz": "Africa/Windhoek",
    },
    "Michotte — Séné": {
        # Editable in the UI: deliberately not treated as a verified precise coordinate.
        "lat": 47.62,
        "lon": -2.74,
        "elev": 10,
        "tz": "Europe/Paris",
    },
}

DEFAULT_TARGETS = [
    {"label": "IC 1848", "mode": "name", "value": "IC 1848"},
    {"label": "LDN 673", "mode": "name", "value": "LDN 673"},
]

if "targets" not in st.session_state:
    st.session_state.targets = DEFAULT_TARGETS.copy()
if "results" not in st.session_state:
    st.session_state.results = None
if "resolved" not in st.session_state:
    st.session_state.resolved = {}


def catalog_name_variants(raw_name):
    """Return sensible Sesame/VizieR name variants for common deep-sky catalogues."""
    raw = re.sub(r"\s+", " ", raw_name.strip())
    compact = re.sub(r"[ _-]+", "", raw).upper()
    variants = [raw]

    rules = [
        (r"^M(\d+)$", lambda n: [f"M {n}", f"M{n}", f"Messier {n}"]),
        (r"^NGC(\d+)$", lambda n: [f"NGC {n}", f"NGC{n}"]),
        (r"^IC(\d+)$", lambda n: [f"IC {n}", f"IC{n}"]),
        (r"^(?:SH2|SHARPLESS)(\d+)$", lambda n: [f"SH 2-{n}", f"Sh2-{n}", f"Sh 2 {n}", f"Sharpless {n}"]),
        (r"^LDN(\d+)$", lambda n: [f"LDN {n}", f"LDN{n}"]),
        (r"^LBN(\d+)$", lambda n: [f"LBN {n}", f"LBN{n}"]),
        (r"^(?:B|BARNARD)(\d+)$", lambda n: [f"Barnard {n}", f"B {n}", f"B{n}"]),
        (r"^(?:ABELL|ACO)(\d+)$", lambda n: [f"Abell {n}", f"ACO {n}"]),
        (r"^(?:SANDQVIST|SANDQ|SA)(\d+)$", lambda n: [f"Sandqvist {n}", f"Sandqvist{n}", f"Sa {n}"]),
        (r"^VDB(\d+)$", lambda n: [f"VDB {n}", f"vdB {n}", f"VDB{n}"]),
        (r"^RCW(\d+)$", lambda n: [f"RCW {n}", f"RCW{n}"]),
        (r"^GUM(\d+)$", lambda n: [f"Gum {n}", f"Gum{n}"]),
        (r"^(?:CED|CEDERBLAD)(\d+)$", lambda n: [f"Ced {n}", f"Cederblad {n}"]),
        (r"^GN(\d+)$", lambda n: [f"GN {n}", f"GN{n}"]),
        (r"^BERNES(\d+)$", lambda n: [f"Bernes {n}", f"Bernes{n}"]),
        (r"^MBM(\d+)$", lambda n: [f"MBM {n}", f"MBM{n}"]),
        (r"^CG(\d+)$", lambda n: [f"CG {n}", f"CG{n}"]),
    ]

    for pattern, maker in rules:
        m = re.match(pattern, compact)
        if m:
            variants.extend(maker(m.group(1)))
            break

    # Keep order while removing duplicates, case-insensitively.
    seen = set()
    out = []
    for name in variants:
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def resolve_target(entry):
    """Resolve a catalogue name robustly, or parse manual RA/DEC."""
    key = (entry["mode"], entry["value"])
    if key in st.session_state.resolved:
        return st.session_state.resolved[key]

    if entry["mode"] == "name":
        attempts = catalog_name_variants(entry["value"])
        last_error = None
        coord = None
        for candidate in attempts:
            try:
                coord = SkyCoord.from_name(candidate)
                break
            except Exception as exc:
                last_error = exc
        if coord is None:
            tried = ", ".join(attempts)
            raise ValueError(f"Cible introuvable. Variantes testées : {tried}") from last_error
    else:
        # Manual syntax stored as "RA|DEC". Accept sexagesimal or decimal degrees.
        ra_txt, dec_txt = [x.strip() for x in entry["value"].split("|", 1)]
        try:
            coord = SkyCoord(float(ra_txt) * u.deg, float(dec_txt) * u.deg, frame="icrs")
        except ValueError:
            coord = SkyCoord(ra_txt, dec_txt, unit=(u.hourangle, u.deg), frame="icrs")

    st.session_state.resolved[key] = coord
    return coord


def local_grid_for_night(day, tz_name, step_min=5):
    """18:00 local on day -> 10:00 local next day, converted to UTC Astropy Time."""
    tz = ZoneInfo(tz_name)
    start_local = datetime(day.year, day.month, day.day, 18, 0, tzinfo=tz)
    end_local = start_local + timedelta(hours=16)
    times_local = []
    t = start_local
    while t <= end_local:
        times_local.append(t)
        t += timedelta(minutes=step_min)
    utc_datetimes = [x.astimezone(timezone.utc) for x in times_local]
    return times_local, Time(utc_datetimes)


def contiguous_windows(mask, times_local, step_min=5):
    windows = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == len(mask) - 1):
            end = i if ok and i == len(mask) - 1 else i - 1
            duration_h = (end - start + 1) * step_min / 60.0
            windows.append((times_local[start], times_local[end] + timedelta(minutes=step_min), duration_h))
            start = None
    return windows


def fmt_h(hours):
    if hours <= 0:
        return "0h00"
    mins = int(round(hours * 60))
    return f"{mins // 60}h{mins % 60:02d}"


FRANCE_TZ = ZoneInfo("Europe/Paris")

def france_time(dt):
    """Convert a timezone-aware local datetime to France time."""
    return dt.astimezone(FRANCE_TZ)

def same_as_france(tz_name):
    return tz_name == "Europe/Paris"


def compute_target_night(coord, location, day, tz_name, min_alt, ha_alert_sep, step_min=5):
    times_local, times = local_grid_for_night(day, tz_name, step_min)
    frame = AltAz(obstime=times, location=location)
    target_altaz = coord.transform_to(frame)
    sun_alt = get_sun(times).transform_to(frame).alt.deg
    moon = get_body("moon", times, location=location)
    moon_alt = moon.transform_to(frame).alt.deg
    sep = coord.separation(moon).deg

    astro = sun_alt < -18.0
    target_ok = target_altaz.alt.deg >= min_alt
    base = astro & target_ok
    moon_down = moon_alt < 0.0
    oiii = base & moon_down
    ha = base
    ha_alert = base & (moon_alt >= 0.0) & (sep <= ha_alert_sep)

    step_h = step_min / 60.0
    base_h = float(base.sum() * step_h)
    oiii_h = float(oiii.sum() * step_h)
    ha_h = float(ha.sum() * step_h)

    if base.any():
        min_sep = float(np.min(sep[base]))
        max_alt = float(np.max(target_altaz.alt.deg[base]))
    else:
        min_sep = math.nan
        max_alt = float(np.max(target_altaz.alt.deg))

    return {
        "date": day,
        "times_local": times_local,
        "target_alt": target_altaz.alt.deg,
        "sun_alt": sun_alt,
        "moon_alt": moon_alt,
        "sep": sep,
        "astro": astro,
        "base": base,
        "oiii": oiii,
        "ha_alert": ha_alert,
        "base_h": base_h,
        "oiii_h": oiii_h,
        "ha_h": ha_h,
        "min_sep": min_sep,
        "max_alt": max_alt,
        "base_windows": contiguous_windows(base, times_local, step_min),
        "oiii_windows": contiguous_windows(oiii, times_local, step_min),
        "ha_alert_any": bool(ha_alert.any()),
    }


def season_trend(nightly):
    """Neutral seasonal trend from usable astronomical-night time over ~1 week."""
    if not nightly:
        return "→ Stable"
    first = nightly[0]["base_h"]
    lookahead = nightly[:min(8, len(nightly))]
    future_max = max(x["base_h"] for x in lookahead)
    ref = lookahead[-1]["base_h"]
    delta = ref - first

    if first < 0.25 and future_max < 0.5:
        return "⛔ Hors saison"
    if delta >= 0.5:
        return "↑ Début de saison"
    if delta <= -0.5:
        return "↓ Fin de saison"
    return "→ Pleine saison"


def culmination_altitude(coord, site_lat):
    """Theoretical meridian culmination altitude from latitude and declination."""
    return 90.0 - abs(float(site_lat) - float(coord.dec.deg))


def build_chart(detail, target_name, show_moon=False):
    x = detail["times_local"]
    target_alt = np.asarray(detail["target_alt"])
    moon_alt = np.asarray(detail["moon_alt"])
    sun_alt = np.asarray(detail["sun_alt"])
    sep = np.asarray(detail["sep"])
    france_labels = [france_time(t).strftime("%H:%M") for t in x]

    fig = go.Figure()

    # Background bands derived directly from solar altitude:
    # day > 0, civil 0/-6, nautical -6/-12, astronomical twilight -12/-18,
    # astronomical night < -18.
    bands = [
        ("Jour", sun_alt >= 0, "rgba(245, 204, 74, 0.28)"),
        ("Crépuscule civil", (sun_alt < 0) & (sun_alt >= -6), "rgba(130, 170, 215, 0.24)"),
        ("Crépuscule nautique", (sun_alt < -6) & (sun_alt >= -12), "rgba(74, 110, 160, 0.28)"),
        ("Crépuscule astro", (sun_alt < -12) & (sun_alt >= -18), "rgba(43, 65, 105, 0.34)"),
        ("Nuit astronomique", sun_alt < -18, "rgba(8, 18, 32, 0.50)"),
    ]

    for label, mask, fill in bands:
        mask = np.asarray(mask, dtype=bool)
        start_i = None
        for i, ok in enumerate(mask):
            if ok and start_i is None:
                start_i = i
            if start_i is not None and (not ok or i == len(mask) - 1):
                end_i = i if ok and i == len(mask) - 1 else i - 1
                x0 = x[start_i]
                x1 = x[min(end_i + 1, len(x) - 1)]
                fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, opacity=1,
                              layer="below", line_width=0)
                start_i = None

    custom_target = np.column_stack((sep, moon_alt, france_labels))
    fig.add_trace(go.Scatter(
        x=x, y=target_alt, mode="lines", name=target_name,
        line=dict(width=4),
        customdata=custom_target,
        hovertemplate=(
            "<b>Site %{x|%H:%M}</b><br>"
            "France %{customdata[2]}<br>"
            "Altitude cible %{y:.1f}°<br>"
            "Altitude Lune %{customdata[1]:.1f}°<br>"
            "Séparation cible-Lune %{customdata[0]:.1f}°"
            "<extra></extra>"
        )
    ))

    if show_moon:
        fig.add_trace(go.Scatter(
            x=x, y=moon_alt, mode="lines", name="Lune",
            line=dict(width=3, dash="dot"),
            customdata=np.column_stack((sep, france_labels)),
            hovertemplate=(
                "<b>Site %{x|%H:%M}</b><br>"
                "France %{customdata[1]}<br>"
                "Altitude Lune %{y:.1f}°<br>"
                "Séparation cible-Lune %{customdata[0]:.1f}°"
                "<extra></extra>"
            )
        ))

    # Transit / maximum altitude within the plotted local-night grid.
    max_i = int(np.argmax(target_alt))
    transit_t = x[max_i]
    transit_alt = float(target_alt[max_i])
    fig.add_vline(x=transit_t, line_dash="dash", line_width=1.5)
    fig.add_annotation(
        x=transit_t, y=min(88, transit_alt + 8),
        text=f"Transit<br><b>{transit_alt:.0f}°</b><br>{transit_t.strftime('%H:%M')}",
        showarrow=False, xanchor="center"
    )

    fig.add_hline(y=20, line_dash="dot", line_width=1, annotation_text="20°")
    fig.update_xaxes(dtick=60 * 60 * 1000, tickformat="%Hh", showgrid=True)
    fig.update_yaxes(dtick=15, showgrid=True, range=[0, 90])

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Heure locale du site",
        yaxis_title="Altitude",
        hovermode="x",
        legend=dict(orientation="h", y=1.10, x=0),
    )
    return fig


st.title("🌌 Astro Planner — V1.5 Beta")
st.caption("Premier regard simple → détail par cible → construction de séquence ensuite.")

with st.sidebar:
    st.header("Observation")
    site_name = st.selectbox("Site", list(SITES.keys()))
    site = SITES[site_name].copy()

    if site_name == "Michotte — Séné":
        st.caption("Coordonnées Michotte modifiables (à valider précisément).")
        site["lat"] = st.number_input("Latitude", value=float(site["lat"]), format="%.6f")
        site["lon"] = st.number_input("Longitude", value=float(site["lon"]), format="%.6f")

    start_day = st.date_input("Première nuit", value=date.today())
    nights = st.slider("Nombre de nuits", 3, 60, 15)
    min_alt = st.slider("Altitude cible minimum (°)", 10, 45, 20)
    ha_alert_sep = st.slider("Alerte Ha — distance Lune", 30, 120, 75,)

    st.divider()
    st.caption("Règles V1")
    st.write("**OIII / LRGB** : Lune sous l’horizon")
    st.write(f"**Ha** : alerte si Lune ≤ {ha_alert_sep}°")
    st.write("**Nuit** : Soleil < −18°")

st.subheader("Cibles")

with st.form("add_name", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    name = c1.text_input(
        "🔎 Rechercher une cible",
        placeholder="M42, NGC 7000, IC 1848, Sh2-173, LDN 673, LBN, Barnard, Abell, Sandqvist…",
        label_visibility="collapsed",
    )
    add_name = c2.form_submit_button("+ Ajouter", use_container_width=True)
    if add_name and name.strip():
        st.session_state.targets.append({"label": name.strip(), "mode": "name", "value": name.strip()})
        st.session_state.results = None
        st.rerun()

with st.expander("⌖ Saisir une cible par coordonnées RA / DEC"):
    st.caption("Format astro — ICRS / J2000")
    with st.form("add_coord", clear_on_submit=True):
        st.markdown("**Ascension droite (RA)**")
        r1, r2, r3 = st.columns(3)
        rah = r1.number_input("h", min_value=0, max_value=23, value=0, step=1)
        ram = r2.number_input("m", min_value=0, max_value=59, value=0, step=1)
        ras = r3.number_input("s", min_value=0.0, max_value=59.999, value=0.0, step=0.1, format="%.1f")
        st.markdown("**Déclinaison (DEC)**")
        d0, d1, d2, d3 = st.columns([0.7, 1, 1, 1])
        sign = d0.selectbox("Signe", ["+", "−"])
        ded = d1.number_input("°", min_value=0, max_value=90, value=0, step=1)
        dem = d2.number_input("′", min_value=0, max_value=59, value=0, step=1)
        des = d3.number_input("″", min_value=0.0, max_value=59.999, value=0.0, step=0.1, format="%.1f")
        custom = st.text_input("Nom de la cible", placeholder="Zone SNR")
        if st.form_submit_button("+ Ajouter cette cible", use_container_width=True):
            ra_txt = f"{int(rah):02d}:{int(ram):02d}:{ras:04.1f}"
            dec_sign = "-" if sign == "−" else "+"
            dec_txt = f"{dec_sign}{int(ded):02d}:{int(dem):02d}:{des:04.1f}"
            label = custom.strip() or f"RA {ra_txt} DEC {dec_txt}"
            st.session_state.targets.append({"label": label, "mode": "coord", "value": f"{ra_txt}|{dec_txt}"})
            st.session_state.results = None
            st.rerun()

if st.session_state.targets:
    st.caption("Cibles sélectionnées")
    cols = st.columns(min(6, len(st.session_state.targets)))
    for i, target in enumerate(st.session_state.targets):
        with cols[i % len(cols)]:
            if st.button(f"{target['label']}  ×", key=f"del_{i}", use_container_width=True):
                st.session_state.targets.pop(i)
                st.session_state.results = None
                st.rerun()
else:
    st.info("Ajoute au moins une cible.")

run = st.button("🔭 Analyser les prochaines nuits", type="primary", use_container_width=True, disabled=not st.session_state.targets)

if run:
    location = EarthLocation(lat=site["lat"] * u.deg, lon=site["lon"] * u.deg, height=site["elev"] * u.m)
    all_results = {}
    errors = []
    progress = st.progress(0, text="Résolution des cibles et calculs…")
    total = len(st.session_state.targets) * nights
    done = 0

    for entry in st.session_state.targets:
        try:
            coord = resolve_target(entry)
            nightly = []
            for n in range(nights):
                d = start_day + timedelta(days=n)
                nightly.append(compute_target_night(coord, location, d, site["tz"], min_alt, ha_alert_sep))
                done += 1
                progress.progress(done / total, text=f"{entry['label']} — {d.strftime('%d/%m')}")
            all_results[entry["label"]] = {"coord": coord, "nightly": nightly}
        except Exception as exc:
            errors.append(f"{entry['label']} : {exc}")
            done += nights

    progress.empty()
    st.session_state.results = {
        "site_name": site_name,
        "site": site,
        "start_day": start_day,
        "nights": nights,
        "min_alt": min_alt,
        "ha_alert_sep": ha_alert_sep,
        "targets": all_results,
        "errors": errors,
    }

res = st.session_state.results
if res:
    if res["errors"]:
        for err in res["errors"]:
            st.warning(err)

    st.divider()
    st.subheader(f"Premier regard — {res['site_name']}")
    summary_rows = []
    for label, data in res["targets"].items():
        tonight = data["nightly"][0]
        total_oiii = sum(x["oiii_h"] for x in data["nightly"])
        total_ha = sum(x["ha_h"] for x in data["nightly"])
        culmin = culmination_altitude(data["coord"], res["site"]["lat"])
        if culmin < 20:
            alt_text = f"⚠ Très basse · {culmin:.0f}° max"
        elif culmin < 30:
            alt_text = f"⚠ Basse · {culmin:.0f}° max"
        else:
            alt_text = f"{tonight['max_alt']:.0f}°"
        summary_rows.append({
            "Cible": label,
            "Saison / tendance": season_trend(data["nightly"]),
            "Visible": fmt_h(tonight["base_h"]),
            "OIII/LRGB": fmt_h(tonight["oiii_h"]),
            "Alt. max": alt_text,
            "Lune min.": "—" if math.isnan(tonight["min_sep"]) else f"{tonight['min_sep']:.0f}°",
            "Ha": "⚠ ≤ seuil" if tonight["ha_alert_any"] else "✓",
            f"OIII {res['nights']} nuits": fmt_h(total_oiii),
            f"Ha {res['nights']} nuits": fmt_h(total_ha),
            "_sort_visible": tonight["base_h"],
        })
    df = pd.DataFrame(summary_rows).sort_values("_sort_visible", ascending=False).drop(columns=["_sort_visible"])

    def highlight_planning(v):
        txt = str(v)
        if "Hors saison" in txt:
            return "background-color: rgba(255, 80, 80, 0.18)"
        if "Fin de saison" in txt:
            return "background-color: rgba(255, 165, 0, 0.18)"
        if "Début de saison" in txt:
            return "background-color: rgba(80, 150, 255, 0.16)"
        if "⚠" in txt:
            return "background-color: rgba(255, 190, 60, 0.18)"
        return ""

    styled = df.style.map(highlight_planning, subset=["Saison / tendance", "Alt. max"])
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.caption("↑ la fenêtre utile augmente · ↓ elle diminue · → période stable. Les avertissements d’altitude signalent une cible structurellement basse depuis ce site.")

    labels = list(res["targets"].keys())
    if labels:
        selected = st.selectbox("🔎 Ouvrir une cible", labels)
        target_data = res["targets"][selected]
        coord = target_data["coord"]
        ra_nina = coord.ra.to_string(unit=u.hour, sep=("h ", "m ", "s"), precision=1, pad=True)
        dec_nina = coord.dec.to_string(unit=u.deg, sep=("° ", "′ ", "″"), precision=0, alwayssign=True, pad=True)
        st.caption(f"RA {ra_nina}  •  DEC {dec_nina}  •  ICRS / J2000")

        detail_rows = []
        for d in target_data["nightly"]:
            detail_rows.append({
                "Nuit": d["date"].strftime("%d/%m/%Y"),
                "Visible ≥ alt min": fmt_h(d["base_h"]),
                "OIII/LRGB": fmt_h(d["oiii_h"]),
                "Ha": fmt_h(d["ha_h"]),
                "Alt. max": f"{d['max_alt']:.0f}°",
                "Distance Lune min.": "—" if math.isnan(d["min_sep"]) else f"{d['min_sep']:.0f}°",
                "Alerte Ha": "⚠️" if d["ha_alert_any"] else "✓",
            })
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)

        night_idx = st.slider("Nuit à afficher", 0, len(target_data["nightly"]) - 1, 0,
                              format="Nuit +%d")
        detail = target_data["nightly"][night_idx]
        st.markdown(f"#### {selected} — nuit du {detail['date'].strftime('%d/%m/%Y')}")
        a, b, c, dcol = st.columns(4)
        a.metric("Visible", fmt_h(detail["base_h"]))
        b.metric("OIII/LRGB", fmt_h(detail["oiii_h"]))
        c.metric("Altitude max", f"{detail['max_alt']:.0f}°")
        dcol.metric("Lune min.", "—" if math.isnan(detail["min_sep"]) else f"{detail['min_sep']:.0f}°")

        if detail["oiii_windows"]:
            if same_as_france(res["site"]["tz"]):
                windows_txt = " • ".join(
                    f"{a.strftime('%H:%M')}→{b.strftime('%H:%M')} ({fmt_h(h)})"
                    for a, b, h in detail["oiii_windows"]
                )
            else:
                windows_txt = " • ".join(
                    f"Site {a.strftime('%H:%M')}→{b.strftime('%H:%M')} | "
                    f"France {france_time(a).strftime('%H:%M')}→{france_time(b).strftime('%H:%M')} "
                    f"({fmt_h(h)})"
                    for a, b, h in detail["oiii_windows"]
                )
            st.success(f"Fenêtre(s) OIII/LRGB sans Lune : {windows_txt}")
        else:
            st.info("Aucune fenêtre OIII/LRGB répondant aux règles cette nuit.")

        moon_key = f"show_moon_{selected}_{night_idx}"
        show_moon = st.toggle("🌙 Afficher la courbe de la Lune", value=False, key=moon_key)
        st.plotly_chart(build_chart(detail, selected, show_moon=show_moon),
                        use_container_width=True, config={"scrollZoom": False})

        # Compact filter availability view.
        st.markdown("##### Disponibilité filtres")
        filter_fig = go.Figure()

        def add_windows(windows, row_name):
            for a0, b0, _h in windows:
                duration_ms = (b0 - a0).total_seconds() * 1000
                midpoint = a0 + (b0 - a0) / 2
                france_a = france_time(a0).strftime("%H:%M")
                france_b = france_time(b0).strftime("%H:%M")
                hover = (
                    f"{row_name}<br>"
                    f"Site {a0.strftime('%H:%M')} → {b0.strftime('%H:%M')}<br>"
                    f"France {france_a} → {france_b}"
                )
                filter_fig.add_trace(go.Bar(
                    x=[duration_ms],
                    y=[row_name],
                    base=[a0],
                    orientation="h",
                    name=row_name,
                    text=[f"{a0.strftime('%H:%M')} → {b0.strftime('%H:%M')}"],
                    textposition="inside",
                    hovertext=[hover],
                    hoverinfo="text",
                    showlegend=False
                ))

        add_windows(detail["base_windows"], "Ha")
        add_windows(detail["oiii_windows"], "OIII / LRGB")

        filter_fig.update_xaxes(
            type="date",
            range=[detail["times_local"][0], detail["times_local"][-1]],
            dtick=60 * 60 * 1000,
            tickformat="%Hh",
            title="Heure locale du site",
            showgrid=True
        )
        filter_fig.update_yaxes(categoryorder="array", categoryarray=["OIII / LRGB", "Ha"])
        filter_fig.update_layout(
            barmode="overlay",
            height=180,
            margin=dict(l=20, r=20, t=5, b=40)
        )
        st.plotly_chart(filter_fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Détail horaire"):
            mask = detail["base"]
            rows = []
            for i, t in enumerate(detail["times_local"]):
                # Calcul remains at 5-minute resolution; display is hourly for readability.
                if mask[i] and t.minute == 0:
                    row = {
                        "Heure site": t.strftime("%H:%M"),
                        "Alt cible": round(float(detail["target_alt"][i]), 1),
                        "Alt Lune": round(float(detail["moon_alt"][i]), 1),
                        "Distance cible-Lune": round(float(detail["sep"][i]), 1),
                        "OIII/LRGB": "✓" if detail["oiii"][i] else "—",
                        "Alerte Ha": "⚠️" if detail["ha_alert"][i] else "—",
                    }
                    if not same_as_france(res["site"]["tz"]):
                        row = {"Heure site": row.pop("Heure site"),
                               "Heure France": france_time(t).strftime("%H:%M"),
                               **row}
                    else:
                        row["Heure"] = row.pop("Heure site")
                    rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.caption("V1 : calcul astronomique uniquement. La météo et le constructeur automatique de séquences viendront après validation de cette base.")
