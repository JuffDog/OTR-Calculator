import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


class MeierOTRModel:
    def predict_otrmax_mol_l_h(
        self,
        flask_diameter_mm,
        filling_volume_ml,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar=1.0,
        o2_fraction=0.2095,
        rpm=250,
    ):
        d_mm = np.asarray(flask_diameter_mm, dtype=float)
        vl_ml = np.asarray(filling_volume_ml, dtype=float)
        d0_cm = np.asarray(shaking_diameter_mm, dtype=float) / 10.0
        osmol = np.asarray(osmolality_osmol_kg, dtype=float)
        n = np.asarray(rpm, dtype=float)
        p_r = np.asarray(pressure_bar, dtype=float)
        y_o2 = np.asarray(o2_fraction, dtype=float)

        return (
            3.72e-7
            * (osmol ** 0.05)
            * (n ** (1.18 - osmol / 10.1))
            * (vl_ml ** -0.74)
            * (d0_cm ** 0.33)
            * (d_mm ** 1.88)
            * p_r
            * y_o2
        )

    def predict_otrmax_mmol_l_h(self, *args, **kwargs):
        return 1000.0 * self.predict_otrmax_mol_l_h(*args, **kwargs)

    def critical_rpm_for_target(
        self,
        target_otr_mmol_l_h,
        flask_diameter_mm,
        filling_volume_ml,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar=1.0,
        o2_fraction=0.2095,
    ):
        if target_otr_mmol_l_h <= 0:
            return None

        d0_cm = shaking_diameter_mm / 10.0
        exponent = 1.18 - osmolality_osmol_kg / 10.1

        prefactor = (
            1000.0
            * 3.72e-7
            * (osmolality_osmol_kg ** 0.05)
            * (filling_volume_ml ** -0.74)
            * (d0_cm ** 0.33)
            * (flask_diameter_mm ** 1.88)
            * pressure_bar
            * o2_fraction
        )

        if prefactor <= 0 or exponent <= 0:
            return None

        return (target_otr_mmol_l_h / prefactor) ** (1.0 / exponent)

    def max_filling_volume_for_target(
        self,
        target_otr_mmol_l_h,
        flask_diameter_mm,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar=1.0,
        o2_fraction=0.2095,
        rpm=250,
    ):
        if target_otr_mmol_l_h <= 0:
            return None

        d0_cm = shaking_diameter_mm / 10.0
        a_without_vl = (
            1000.0
            * 3.72e-7
            * (osmolality_osmol_kg ** 0.05)
            * (rpm ** (1.18 - osmolality_osmol_kg / 10.1))
            * (d0_cm ** 0.33)
            * (flask_diameter_mm ** 1.88)
            * pressure_bar
            * o2_fraction
        )

        if a_without_vl <= 0:
            return None

        return (a_without_vl / target_otr_mmol_l_h) ** (1.0 / 0.74)


# Kühner OTR calculator defaults / convenience values
FLASK_DIAMETERS = {
    "125 mL Erlenmeyer — 65.5 mm": 65.5,
    "250 mL Erlenmeyer — 83.3 mm": 83.3,
    "500 mL Erlenmeyer — 101.3 mm": 101.3,
    "1000 mL Erlenmeyer — 127.8 mm": 127.8,
    "Custom/manual entry": None,
}

ORBIT_OPTIONS = {
    "12.5 mm": 12.5,
    "19 mm": 19.0,
    "25 mm": 25.0,
    "50 mm": 50.0,
    "Custom/manual entry": None,
}

# Medium osmolality defaults from the Kühner OTR calculator.
MEDIUM_OSMOLALITIES = {
    "LB — 0.240 osmol/kg": 0.240,
    "YPD glycerol — 0.250 osmol/kg": 0.250,
    "EXPI293 — 0.285 osmol/kg": 0.285,
    "EXPI CHO — 0.295 osmol/kg": 0.295,
    "YPD glucose — 0.350 osmol/kg": 0.350,
    "TB — 0.500 osmol/kg": 0.500,
    "SYN6-MES — 0.660 osmol/kg": 0.660,
    "Custom/manual entry": None,
}

STANDARD_PRESSURE_BAR = 1.0
STANDARD_O2_FRACTION = 0.2095
EXPECTED_ERROR_MMOL_L_H = 5.0


def flask_diameter_input(label, key_prefix, default_choice="250 mL Erlenmeyer — 83.3 mm"):
    choice = st.selectbox(label, list(FLASK_DIAMETERS.keys()), index=list(FLASK_DIAMETERS.keys()).index(default_choice), key=f"{key_prefix}_flask_choice")
    value = FLASK_DIAMETERS[choice]
    if value is None:
        return st.number_input("Custom flask maximum diameter d [mm]", min_value=30.0, max_value=200.0, value=83.3, step=0.1, key=f"{key_prefix}_flask_custom")
    st.caption(f"Diameter used in equation: {value:.1f} mm")
    return value


def orbit_input(label, key_prefix, default_choice="50 mm"):
    choice = st.selectbox(label, list(ORBIT_OPTIONS.keys()), index=list(ORBIT_OPTIONS.keys()).index(default_choice), key=f"{key_prefix}_orbit_choice")
    value = ORBIT_OPTIONS[choice]
    if value is None:
        return st.number_input("Custom shaking diameter d₀ [mm]", min_value=5.0, max_value=100.0, value=50.0, step=0.1, key=f"{key_prefix}_orbit_custom")
    return value


def osmolality_input(label, key_prefix, default_choice="YPD glucose — 0.350 osmol/kg"):
    choice = st.selectbox(label, list(MEDIUM_OSMOLALITIES.keys()), index=list(MEDIUM_OSMOLALITIES.keys()).index(default_choice), key=f"{key_prefix}_medium_choice")
    value = MEDIUM_OSMOLALITIES[choice]
    if value is None:
        return st.number_input("Custom media osmolality [osmol/kg]", min_value=0.01, max_value=2.50, value=0.30, step=0.01, key=f"{key_prefix}_osmo_custom")
    return value


def validation_status(rpm, filling_volume_ml, shaking_diameter_mm, flask_diameter_mm):
    checks = {
        "Shaking frequency n": 100 <= rpm <= 450,
        "Filling volume V_L": 2 <= filling_volume_ml <= 160,
        "Shaking diameter d₀": 12.5 <= shaking_diameter_mm <= 100,
        "Flask diameter d": 51 <= flask_diameter_mm <= 131,
    }
    n_ok = sum(checks.values())
    if n_ok == 4:
        return "High confidence", "Inputs are inside the broad validation range reported in the paper.", checks
    elif n_ok >= 2:
        return "Moderate extrapolation", "Some inputs are outside the reported validation range.", checks
    return "Strong extrapolation", "Several inputs are outside the reported validation range. Treat result as rough estimate only.", checks


def show_validation(status, message, checks):
    if status == "High confidence":
        st.success(f"🟢 {status}: {message}")
    elif status == "Moderate extrapolation":
        st.warning(f"🟡 {status}: {message}")
    else:
        st.error(f"🔴 {status}: {message}")

    with st.expander("Validation range details", expanded=False):
        ranges = {
            "Shaking frequency n": "100–450 rpm",
            "Filling volume V_L": "2–160 mL",
            "Shaking diameter d₀": "12.5–100 mm",
            "Flask diameter d": "51–131 mm",
        }
        for key, ok in checks.items():
            symbol = "✅" if ok else "⚠️"
            st.write(f"{symbol} {key}: {ranges[key]}")


model = MeierOTRModel()

st.set_page_config(page_title="OTRmax Calculator — Paper Model", page_icon="🧪", layout="wide")
st.title("🧪 OTRmax Calculator for Non-Baffled Shake Flasks")
st.caption("Paper-based empirical correlation after Meier et al. 2016; convenience flask/orbit/medium defaults aligned with Kühner's OTR calculator.")

with st.expander("Model equation and units", expanded=False):
    st.markdown(
        """
        **Original paper equation**

        $$
        OTR_{max} =
        3.72 \\times 10^{-7}
        \\cdot Osmol^{0.05}
        \\cdot n^{(1.18 - Osmol/10.1)}
        \\cdot V_L^{-0.74}
        \\cdot d_0^{0.33}
        \\cdot d^{1.88}
        \\cdot p_R
        \\cdot y_{O_2}
        $$

        The original equation returns **mol L⁻¹ h⁻¹**. This app converts to **mmol L⁻¹ h⁻¹**.

        Units: $Osmol$ in osmol/kg, $n$ in rpm, $V_L$ in mL, $d_0$ in cm internally, $d$ in mm, $p_R$ in bar abs., $y_{O_2}$ as mole fraction.
        """
    )

tab_calc, tab_heatmap, tab_optimizer, tab_compare = st.tabs(
    ["Calculator", "Design-space heatmap", "Target solver", "Compare setups"]
)

with tab_calc:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input parameters")
        flask_diameter_mm = flask_diameter_input("Shake flask maximum diameter d [mm]", "calc")
        filling_volume_ml = st.number_input("Filling volume V_L [mL]", min_value=1.0, max_value=250.0, value=25.0, step=1.0, key="calc_vl")
        shaking_diameter_mm = orbit_input("Shaking diameter d₀ / orbit [mm]", "calc")
        osmolality_osmol_kg = osmolality_input("Medium / osmolality [osmol/kg]", "calc")
        rpm = st.number_input("Shaking frequency n [rpm]", min_value=50.0, max_value=600.0, value=250.0, step=5.0, key="calc_rpm")
        target_or_our = st.number_input("Optional target OUR / OTR demand [mmol/L/h]", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="calc_our")

        with st.expander("Advanced settings: gas phase", expanded=False):
            pressure_bar = st.number_input("Pressure p_R [bar abs.]", min_value=0.1, max_value=5.0, value=STANDARD_PRESSURE_BAR, step=0.1, key="calc_pressure")
            o2_fraction = st.number_input("O₂ mole fraction y_O₂ [-]", min_value=0.01, max_value=1.0, value=STANDARD_O2_FRACTION, step=0.0005, format="%.4f", key="calc_o2")

    with col2:
        st.subheader("Calculated result")
        st.caption(f"Gas phase used: p_R = {pressure_bar:.3f} bar abs., y_O₂ = {o2_fraction:.4f}")

        otrmax = model.predict_otrmax_mmol_l_h(flask_diameter_mm, filling_volume_ml, shaking_diameter_mm, osmolality_osmol_kg, pressure_bar, o2_fraction, rpm)
        st.metric("Predicted OTRmax", f"{otrmax:.2f} mmol/L/h")
        lower = max(0.0, otrmax - EXPECTED_ERROR_MMOL_L_H)
        upper = otrmax + EXPECTED_ERROR_MMOL_L_H
        st.caption(f"Approx. paper-level uncertainty band: {lower:.2f}–{upper:.2f} mmol/L/h")

        if target_or_our > 0:
            utilization = 100.0 * target_or_our / otrmax if otrmax > 0 else np.inf
            st.metric("OTR utilization", f"{utilization:.0f} %")
            if utilization < 70:
                st.success("🟢 Oxygen transfer reserve appears comfortable.")
            elif utilization < 90:
                st.warning("🟡 Oxygen transfer reserve is moderate.")
            else:
                st.error("🔴 Oxygen limitation risk or very low reserve.")
            crit_rpm = model.critical_rpm_for_target(target_or_our, flask_diameter_mm, filling_volume_ml, shaking_diameter_mm, osmolality_osmol_kg, pressure_bar, o2_fraction)
            if crit_rpm is not None:
                st.metric("Minimum rpm for target demand", f"{crit_rpm:.0f} rpm")

        status, message, checks = validation_status(rpm, filling_volume_ml, shaking_diameter_mm, flask_diameter_mm)
        show_validation(status, message, checks)

    st.subheader("OTRmax vs shaking frequency")
    rpm_range = np.linspace(50, 600, 150)
    otr_values = [model.predict_otrmax_mmol_l_h(flask_diameter_mm, filling_volume_ml, shaking_diameter_mm, osmolality_osmol_kg, pressure_bar, o2_fraction, r) for r in rpm_range]
    fig, ax = plt.subplots()
    ax.plot(rpm_range, otr_values)
    ax.axvline(rpm, linestyle="--", label="selected rpm")
    if target_or_our > 0:
        ax.axhline(target_or_our, linestyle=":", label="target demand")
    ax.set_xlabel("Shaking frequency n [rpm]")
    ax.set_ylabel("Predicted OTRmax [mmol/L/h]")
    ax.set_title("OTRmax vs shaking frequency")
    ax.legend()
    st.pyplot(fig)

    export_df = pd.DataFrame([{
        "shake_flask_diameter_d_mm": flask_diameter_mm,
        "filling_volume_VL_ml": filling_volume_ml,
        "shaking_diameter_d0_mm": shaking_diameter_mm,
        "shaking_diameter_d0_cm_used_in_equation": shaking_diameter_mm / 10.0,
        "media_osmolality_osmol_kg": osmolality_osmol_kg,
        "pressure_pR_bar_abs": pressure_bar,
        "o2_mole_fraction_yO2": o2_fraction,
        "shaking_frequency_n_rpm": rpm,
        "target_demand_mmol_L_h": target_or_our,
        "predicted_otrmax_mmol_L_h": otrmax,
        "expected_lower_mmol_L_h": lower,
        "expected_upper_mmol_L_h": upper,
        "confidence": status,
    }])
    st.download_button("Download current calculation as CSV", export_df.to_csv(index=False).encode("utf-8"), file_name="otrmax_paper_model_result.csv", mime="text/csv")

with tab_heatmap:
    st.subheader("Design-space heatmap: OTRmax over rpm × filling volume")
    colh1, colh2 = st.columns([1, 1])
    with colh1:
        hm_d = flask_diameter_input("Shake flask maximum diameter d [mm]", "hm")
        hm_d0 = orbit_input("Shaking diameter d₀ / orbit [mm]", "hm")
        hm_osmo = osmolality_input("Medium / osmolality [osmol/kg]", "hm")
    with colh2:
        hm_rpm_min, hm_rpm_max = st.slider("RPM range", min_value=50, max_value=600, value=(100, 450), step=10, key="hm_rpm_range")
        hm_vl_min, hm_vl_max = st.slider("Filling volume range [mL]", min_value=1, max_value=250, value=(2, 160), step=1, key="hm_vl_range")
        with st.expander("Advanced settings: gas phase", expanded=False):
            hm_pressure = st.number_input("Pressure p_R [bar abs.]", min_value=0.1, max_value=5.0, value=STANDARD_PRESSURE_BAR, step=0.1, key="hm_pressure")
            hm_o2 = st.number_input("O₂ mole fraction y_O₂ [-]", min_value=0.01, max_value=1.0, value=STANDARD_O2_FRACTION, step=0.0005, format="%.4f", key="hm_o2")

    rpm_grid = np.linspace(hm_rpm_min, hm_rpm_max, 90)
    vl_grid = np.linspace(hm_vl_min, hm_vl_max, 90)
    R, V = np.meshgrid(rpm_grid, vl_grid)
    Z = model.predict_otrmax_mmol_l_h(hm_d, V, hm_d0, hm_osmo, hm_pressure, hm_o2, R)

    fig2, ax2 = plt.subplots()
    contour = ax2.contourf(R, V, Z, levels=24)
    fig2.colorbar(contour, ax=ax2, label="OTRmax [mmol/L/h]")
    ax2.set_xlabel("Shaking frequency n [rpm]")
    ax2.set_ylabel("Filling volume V_L [mL]")
    ax2.set_title("OTRmax design space")
    st.pyplot(fig2)

with tab_optimizer:
    st.subheader("Target solver")
    col_o1, col_o2 = st.columns([1, 1])
    with col_o1:
        opt_target = st.number_input("Target OTRmax / oxygen demand [mmol/L/h]", min_value=1.0, max_value=500.0, value=100.0, step=5.0, key="opt_target")
        opt_d = flask_diameter_input("Shake flask maximum diameter d [mm]", "opt")
        opt_vl = st.number_input("Filling volume V_L [mL]", min_value=1.0, max_value=250.0, value=25.0, step=1.0, key="opt_vl")
        opt_d0 = orbit_input("Shaking diameter d₀ / orbit [mm]", "opt")
    with col_o2:
        opt_osmo = osmolality_input("Medium / osmolality [osmol/kg]", "opt")
        opt_rpm = st.number_input("Available / current shaking frequency [rpm]", min_value=50.0, max_value=600.0, value=250.0, step=5.0, key="opt_rpm")
        with st.expander("Advanced settings: gas phase", expanded=False):
            opt_pressure = st.number_input("Pressure p_R [bar abs.]", min_value=0.1, max_value=5.0, value=STANDARD_PRESSURE_BAR, step=0.1, key="opt_pressure")
            opt_o2 = st.number_input("O₂ mole fraction y_O₂ [-]", min_value=0.01, max_value=1.0, value=STANDARD_O2_FRACTION, step=0.0005, format="%.4f", key="opt_o2")

    current_otr = model.predict_otrmax_mmol_l_h(opt_d, opt_vl, opt_d0, opt_osmo, opt_pressure, opt_o2, opt_rpm)
    required_rpm = model.critical_rpm_for_target(opt_target, opt_d, opt_vl, opt_d0, opt_osmo, opt_pressure, opt_o2)
    max_volume = model.max_filling_volume_for_target(opt_target, opt_d, opt_d0, opt_osmo, opt_pressure, opt_o2, opt_rpm)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current OTRmax", f"{current_otr:.1f} mmol/L/h")
    if required_rpm is not None:
        c2.metric("Required rpm at current filling volume", f"{required_rpm:.0f} rpm")
    if max_volume is not None:
        c3.metric("Max. filling volume at current rpm", f"{max_volume:.1f} mL")

    st.markdown("### Automatic search for feasible operating points")
    search_rpm_min, search_rpm_max = st.slider("Search rpm range", min_value=50, max_value=600, value=(100, 450), step=10, key="opt_search_rpm")
    search_vl_min, search_vl_max = st.slider("Search filling volume range [mL]", min_value=1, max_value=250, value=(2, 160), step=1, key="opt_search_vl")

    candidates = []
    for candidate_rpm in np.arange(search_rpm_min, search_rpm_max + 1, 5):
        for candidate_vl in np.arange(search_vl_min, search_vl_max + 1, 1):
            candidate_otr = model.predict_otrmax_mmol_l_h(opt_d, candidate_vl, opt_d0, opt_osmo, opt_pressure, opt_o2, candidate_rpm)
            if candidate_otr >= opt_target:
                candidates.append({"rpm": candidate_rpm, "filling_volume_ml": candidate_vl, "predicted_otrmax_mmol_L_h": candidate_otr, "reserve_mmol_L_h": candidate_otr - opt_target})

    if candidates:
        cand_df = pd.DataFrame(candidates).sort_values(by=["rpm", "reserve_mmol_L_h", "filling_volume_ml"], ascending=[True, True, False]).head(20)
        st.dataframe(cand_df, use_container_width=True)
        st.download_button("Download feasible operating points as CSV", cand_df.to_csv(index=False).encode("utf-8"), file_name="otrmax_feasible_operating_points.csv", mime="text/csv")
    else:
        st.error("No feasible operating points found in the selected search range.")

with tab_compare:
    st.subheader("Compare multiple setups")
    num_setups = st.slider("Number of setups", 2, 6, 3)
    rows = []

    for i in range(num_setups):
        st.markdown(f"### Setup {i + 1}")
        c1, c2, c3 = st.columns(3)
        with c1:
            setup_name = st.text_input("Setup name", value=f"Setup {i + 1}", key=f"cmp_name_{i}")
            cmp_d = flask_diameter_input("d [mm]", f"cmp_{i}")
            cmp_vl = st.number_input("V_L [mL]", min_value=1.0, max_value=250.0, value=25.0, step=1.0, key=f"cmp_vl_{i}")
        with c2:
            cmp_d0 = orbit_input("d₀ / orbit [mm]", f"cmp_{i}")
            cmp_osmo = osmolality_input("Medium / osmolality [osmol/kg]", f"cmp_{i}")
            cmp_rpm = st.number_input("rpm", min_value=50.0, max_value=600.0, value=250.0, step=5.0, key=f"cmp_rpm_{i}")
        with c3:
            with st.expander("Advanced gas phase", expanded=False):
                cmp_pressure = st.number_input("p_R [bar abs.]", min_value=0.1, max_value=5.0, value=STANDARD_PRESSURE_BAR, step=0.1, key=f"cmp_pressure_{i}")
                cmp_o2 = st.number_input("y_O₂ [-]", min_value=0.01, max_value=1.0, value=STANDARD_O2_FRACTION, step=0.0005, format="%.4f", key=f"cmp_o2_{i}")

        cmp_otr = model.predict_otrmax_mmol_l_h(cmp_d, cmp_vl, cmp_d0, cmp_osmo, cmp_pressure, cmp_o2, cmp_rpm)
        cmp_status, _, _ = validation_status(cmp_rpm, cmp_vl, cmp_d0, cmp_d)
        rows.append({"Setup": setup_name, "d [mm]": cmp_d, "V_L [mL]": cmp_vl, "d0 [mm]": cmp_d0, "Osmol [osmol/kg]": cmp_osmo, "rpm": cmp_rpm, "p_R [bar abs.]": cmp_pressure, "y_O2 [-]": cmp_o2, "OTRmax [mmol/L/h]": cmp_otr, "Expected lower": max(0.0, cmp_otr - EXPECTED_ERROR_MMOL_L_H), "Expected upper": cmp_otr + EXPECTED_ERROR_MMOL_L_H, "Confidence": cmp_status})

    comparison_df = pd.DataFrame(rows)
    st.markdown("## Comparison result")
    st.dataframe(comparison_df, use_container_width=True)

    fig3, ax3 = plt.subplots()
    ax3.bar(comparison_df["Setup"], comparison_df["OTRmax [mmol/L/h]"])
    ax3.set_ylabel("OTRmax [mmol/L/h]")
    ax3.set_title("Setup comparison")
    ax3.tick_params(axis="x", rotation=30)
    st.pyplot(fig3)

    st.download_button("Download comparison as CSV", comparison_df.to_csv(index=False).encode("utf-8"), file_name="otrmax_setup_comparison.csv", mime="text/csv")
