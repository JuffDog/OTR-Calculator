import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# OTRmax Calculator for non-baffled shake flasks
# Paper model: Meier et al. 2016, Biochemical Engineering Journal
#
# Original paper equation returns OTRmax in mol/L/h:
#
# OTRmax = 3.72e-7
#          * Osmol^0.05
#          * n^(1.18 - Osmol/10.1)
#          * V_L^-0.74
#          * d0^0.33
#          * d^1.88
#          * p_R
#          * y_O2
#
# Units:
# - OTRmax: mol/L/h in original equation
# - Osmol: osmol/kg
# - n: rpm
# - V_L: mL
# - d0: cm
# - d: mm
# - p_R: bar absolute
# - y_O2: O2 mole fraction
#
# This app reports OTRmax in mmol/L/h by multiplying by 1000.
# ============================================================


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

        d_mm = flask_diameter_mm
        vl_ml = filling_volume_ml
        d0_cm = shaking_diameter_mm / 10.0
        osmol = osmolality_osmol_kg
        p_r = pressure_bar
        y_o2 = o2_fraction

        exponent = 1.18 - osmol / 10.1

        # mmol/L/h prefactor, because target is mmol/L/h
        prefactor = (
            1000.0
            * 3.72e-7
            * (osmol ** 0.05)
            * (vl_ml ** -0.74)
            * (d0_cm ** 0.33)
            * (d_mm ** 1.88)
            * p_r
            * y_o2
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

        d_mm = flask_diameter_mm
        d0_cm = shaking_diameter_mm / 10.0
        osmol = osmolality_osmol_kg
        n = rpm
        p_r = pressure_bar
        y_o2 = o2_fraction

        # OTR = A * V_L^-0.74
        a_without_vl = (
            1000.0
            * 3.72e-7
            * (osmol ** 0.05)
            * (n ** (1.18 - osmol / 10.1))
            * (d0_cm ** 0.33)
            * (d_mm ** 1.88)
            * p_r
            * y_o2
        )

        if a_without_vl <= 0:
            return None

        return (a_without_vl / target_otr_mmol_l_h) ** (1.0 / 0.74)


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
    else:
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


def calculate_comparison(model, setup):
    return model.predict_otrmax_mmol_l_h(
        setup["flask_diameter_mm"],
        setup["filling_volume_ml"],
        setup["shaking_diameter_mm"],
        setup["osmolality_osmol_kg"],
        setup["pressure_bar"],
        setup["o2_fraction"],
        setup["rpm"],
    )


model = MeierOTRModel()

STANDARD_PRESSURE_BAR = 1.0
STANDARD_O2_FRACTION = 0.2095
EXPECTED_ERROR_MMOL_L_H = 5.0

st.set_page_config(
    page_title="OTRmax Calculator — Paper Model",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 OTRmax Calculator for Non-Baffled Shake Flasks")
st.caption("Paper-based empirical correlation after Meier et al. 2016.")

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

        The original equation returns **mol L⁻¹ h⁻¹**.  
        This app converts the result to **mmol L⁻¹ h⁻¹** by multiplying by 1000.

        **Units used in the equation**

        - $Osmol$: osmol kg⁻¹
        - $n$: rpm
        - $V_L$: mL
        - $d_0$: cm — this app accepts mm and converts to cm internally
        - $d$: mm
        - $p_R$: bar absolute
        - $y_{O_2}$: O₂ mole fraction, e.g. 0.2095 for ambient air
        """
    )

st.sidebar.header("Presets")

flask_preset = st.sidebar.selectbox(
    "Flask preset",
    ["Custom", "100 mL Erlenmeyer", "250 mL Erlenmeyer", "500 mL Erlenmeyer", "1000 mL Erlenmeyer"],
)

preset_values = {
    "Custom": (80.0, 50.0),
    "100 mL Erlenmeyer": (64.0, 10.0),
    "250 mL Erlenmeyer": (85.0, 25.0),
    "500 mL Erlenmeyer": (105.0, 50.0),
    "1000 mL Erlenmeyer": (131.0, 100.0),
}

default_diameter, default_volume = preset_values[flask_preset]

tab_calc, tab_heatmap, tab_optimizer, tab_compare = st.tabs(
    ["Calculator", "Design-space heatmap", "Target solver", "Compare setups"]
)

with tab_calc:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input parameters")

        flask_diameter_mm = st.number_input(
            "Shake flask maximum diameter d [mm]",
            min_value=30.0,
            max_value=200.0,
            value=default_diameter,
            step=1.0,
            key="calc_d",
        )

        filling_volume_ml = st.number_input(
            "Filling volume V_L [mL]",
            min_value=1.0,
            max_value=250.0,
            value=default_volume,
            step=1.0,
            key="calc_vl",
        )

        shaking_diameter_mm = st.number_input(
            "Shaking diameter d₀ [mm]",
            min_value=5.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            key="calc_d0",
        )

        osmolality_osmol_kg = st.number_input(
            "Media osmolality [osmol/kg]",
            min_value=0.01,
            max_value=2.50,
            value=0.30,
            step=0.01,
            key="calc_osmo",
        )

        rpm = st.number_input(
            "Shaking frequency n [rpm]",
            min_value=50.0,
            max_value=600.0,
            value=250.0,
            step=5.0,
            key="calc_rpm",
        )

        target_or_our = st.number_input(
            "Optional target OUR / OTR demand [mmol/L/h]",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=1.0,
            key="calc_our",
        )

        with st.expander("Advanced settings: gas phase", expanded=False):
            st.caption("Default corresponds approximately to ambient air at 1 bar absolute.")
            pressure_bar = st.number_input(
                "Pressure p_R [bar abs.]",
                min_value=0.1,
                max_value=5.0,
                value=STANDARD_PRESSURE_BAR,
                step=0.1,
                key="calc_pressure",
            )

            o2_fraction = st.number_input(
                "O₂ mole fraction y_O₂ [-]",
                min_value=0.01,
                max_value=1.0,
                value=STANDARD_O2_FRACTION,
                step=0.0005,
                format="%.4f",
                key="calc_o2",
            )

    with col2:
        st.subheader("Calculated result")
        st.caption(f"Gas phase used: p_R = {pressure_bar:.3f} bar abs., y_O₂ = {o2_fraction:.4f}")

        otrmax = model.predict_otrmax_mmol_l_h(
            flask_diameter_mm,
            filling_volume_ml,
            shaking_diameter_mm,
            osmolality_osmol_kg,
            pressure_bar,
            o2_fraction,
            rpm,
        )

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

            crit_rpm = model.critical_rpm_for_target(
                target_or_our,
                flask_diameter_mm,
                filling_volume_ml,
                shaking_diameter_mm,
                osmolality_osmol_kg,
                pressure_bar,
                o2_fraction,
            )

            if crit_rpm is not None:
                st.metric("Minimum rpm for target demand", f"{crit_rpm:.0f} rpm")

        status, message, checks = validation_status(
            rpm, filling_volume_ml, shaking_diameter_mm, flask_diameter_mm
        )
        show_validation(status, message, checks)

    st.subheader("OTRmax vs shaking frequency")

    rpm_range = np.linspace(50, 600, 150)
    otr_values = [
        model.predict_otrmax_mmol_l_h(
            flask_diameter_mm,
            filling_volume_ml,
            shaking_diameter_mm,
            osmolality_osmol_kg,
            pressure_bar,
            o2_fraction,
            r,
        )
        for r in rpm_range
    ]

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

    export_df = pd.DataFrame(
        [
            {
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
            }
        ]
    )

    st.download_button(
        "Download current calculation as CSV",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name="otrmax_paper_model_result.csv",
        mime="text/csv",
    )

with tab_heatmap:
    st.subheader("Design-space heatmap: OTRmax over rpm × filling volume")

    colh1, colh2 = st.columns([1, 1])

    with colh1:
        hm_d = st.number_input(
            "Shake flask maximum diameter d [mm]",
            min_value=30.0,
            max_value=200.0,
            value=default_diameter,
            step=1.0,
            key="hm_d",
        )
        hm_d0 = st.number_input(
            "Shaking diameter d₀ [mm]",
            min_value=5.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            key="hm_d0",
        )
        hm_osmo = st.number_input(
            "Media osmolality [osmol/kg]",
            min_value=0.01,
            max_value=2.5,
            value=0.30,
            step=0.01,
            key="hm_osmo",
        )

    with colh2:
        hm_rpm_min, hm_rpm_max = st.slider(
            "RPM range",
            min_value=50,
            max_value=600,
            value=(100, 450),
            step=10,
            key="hm_rpm_range",
        )
        hm_vl_min, hm_vl_max = st.slider(
            "Filling volume range [mL]",
            min_value=1,
            max_value=250,
            value=(2, 160),
            step=1,
            key="hm_vl_range",
        )
        with st.expander("Advanced settings: gas phase", expanded=False):
            hm_pressure = st.number_input(
                "Pressure p_R [bar abs.]",
                min_value=0.1,
                max_value=5.0,
                value=STANDARD_PRESSURE_BAR,
                step=0.1,
                key="hm_pressure",
            )
            hm_o2 = st.number_input(
                "O₂ mole fraction y_O₂ [-]",
                min_value=0.01,
                max_value=1.0,
                value=STANDARD_O2_FRACTION,
                step=0.0005,
                format="%.4f",
                key="hm_o2",
            )

    rpm_grid = np.linspace(hm_rpm_min, hm_rpm_max, 90)
    vl_grid = np.linspace(hm_vl_min, hm_vl_max, 90)
    R, V = np.meshgrid(rpm_grid, vl_grid)

    Z = model.predict_otrmax_mmol_l_h(
        hm_d,
        V,
        hm_d0,
        hm_osmo,
        hm_pressure,
        hm_o2,
        R,
    )

    fig2, ax2 = plt.subplots()
    contour = ax2.contourf(R, V, Z, levels=24)
    fig2.colorbar(contour, ax=ax2, label="OTRmax [mmol/L/h]")
    ax2.set_xlabel("Shaking frequency n [rpm]")
    ax2.set_ylabel("Filling volume V_L [mL]")
    ax2.set_title("OTRmax design space")
    st.pyplot(fig2)

    st.caption("Tip: High OTRmax regions are reached by increasing rpm and/or reducing filling volume.")

with tab_optimizer:
    st.subheader("Target solver")

    col_o1, col_o2 = st.columns([1, 1])

    with col_o1:
        opt_target = st.number_input(
            "Target OTRmax / oxygen demand [mmol/L/h]",
            min_value=1.0,
            max_value=500.0,
            value=100.0,
            step=5.0,
            key="opt_target",
        )
        opt_d = st.number_input(
            "Shake flask maximum diameter d [mm]",
            min_value=30.0,
            max_value=200.0,
            value=default_diameter,
            step=1.0,
            key="opt_d",
        )
        opt_vl = st.number_input(
            "Filling volume V_L [mL]",
            min_value=1.0,
            max_value=250.0,
            value=default_volume,
            step=1.0,
            key="opt_vl",
        )
        opt_d0 = st.number_input(
            "Shaking diameter d₀ [mm]",
            min_value=5.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            key="opt_d0",
        )

    with col_o2:
        opt_osmo = st.number_input(
            "Media osmolality [osmol/kg]",
            min_value=0.01,
            max_value=2.5,
            value=0.30,
            step=0.01,
            key="opt_osmo",
        )
        opt_rpm = st.number_input(
            "Available / current shaking frequency [rpm]",
            min_value=50.0,
            max_value=600.0,
            value=250.0,
            step=5.0,
            key="opt_rpm",
        )
        with st.expander("Advanced settings: gas phase", expanded=False):
            opt_pressure = st.number_input(
                "Pressure p_R [bar abs.]",
                min_value=0.1,
                max_value=5.0,
                value=STANDARD_PRESSURE_BAR,
                step=0.1,
                key="opt_pressure",
            )
            opt_o2 = st.number_input(
                "O₂ mole fraction y_O₂ [-]",
                min_value=0.01,
                max_value=1.0,
                value=STANDARD_O2_FRACTION,
                step=0.0005,
                format="%.4f",
                key="opt_o2",
            )

    current_otr = model.predict_otrmax_mmol_l_h(
        opt_d, opt_vl, opt_d0, opt_osmo, opt_pressure, opt_o2, opt_rpm
    )

    required_rpm = model.critical_rpm_for_target(
        opt_target, opt_d, opt_vl, opt_d0, opt_osmo, opt_pressure, opt_o2
    )

    max_volume = model.max_filling_volume_for_target(
        opt_target, opt_d, opt_d0, opt_osmo, opt_pressure, opt_o2, opt_rpm
    )

    st.markdown("### Solver results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current OTRmax", f"{current_otr:.1f} mmol/L/h")

    if required_rpm is not None:
        c2.metric("Required rpm at current filling volume", f"{required_rpm:.0f} rpm")
        if required_rpm > 450:
            c2.warning("Above validated rpm range.")
    else:
        c2.error("Could not solve required rpm.")

    if max_volume is not None:
        c3.metric("Max. filling volume at current rpm", f"{max_volume:.1f} mL")
        if max_volume > 160:
            c3.warning("Above validated filling-volume range.")
    else:
        c3.error("Could not solve max. filling volume.")

    st.markdown("### Automatic search for feasible operating points")

    search_rpm_min, search_rpm_max = st.slider(
        "Search rpm range",
        min_value=50,
        max_value=600,
        value=(100, 450),
        step=10,
        key="opt_search_rpm",
    )
    search_vl_min, search_vl_max = st.slider(
        "Search filling volume range [mL]",
        min_value=1,
        max_value=250,
        value=(2, 160),
        step=1,
        key="opt_search_vl",
    )

    candidates = []
    for candidate_rpm in np.arange(search_rpm_min, search_rpm_max + 1, 5):
        for candidate_vl in np.arange(search_vl_min, search_vl_max + 1, 1):
            candidate_otr = model.predict_otrmax_mmol_l_h(
                opt_d,
                candidate_vl,
                opt_d0,
                opt_osmo,
                opt_pressure,
                opt_o2,
                candidate_rpm,
            )
            if candidate_otr >= opt_target:
                candidates.append(
                    {
                        "rpm": candidate_rpm,
                        "filling_volume_ml": candidate_vl,
                        "predicted_otrmax_mmol_L_h": candidate_otr,
                        "reserve_mmol_L_h": candidate_otr - opt_target,
                    }
                )

    if candidates:
        cand_df = pd.DataFrame(candidates)
        cand_df = cand_df.sort_values(
            by=["rpm", "reserve_mmol_L_h", "filling_volume_ml"],
            ascending=[True, True, False],
        ).head(20)
        st.dataframe(cand_df, use_container_width=True)
        st.download_button(
            "Download feasible operating points as CSV",
            cand_df.to_csv(index=False).encode("utf-8"),
            file_name="otrmax_feasible_operating_points.csv",
            mime="text/csv",
        )
    else:
        st.error("No feasible operating points found in the selected search range.")

with tab_compare:
    st.subheader("Compare multiple setups")

    st.caption("Use this to compare different flask sizes, filling volumes, shaking diameters or rpm settings.")

    num_setups = st.slider("Number of setups", 2, 6, 3)

    rows = []

    for i in range(num_setups):
        st.markdown(f"### Setup {i + 1}")
        c1, c2, c3 = st.columns(3)

        with c1:
            setup_name = st.text_input("Setup name", value=f"Setup {i + 1}", key=f"cmp_name_{i}")
            cmp_d = st.number_input(
                "d [mm]",
                min_value=30.0,
                max_value=200.0,
                value=default_diameter,
                step=1.0,
                key=f"cmp_d_{i}",
            )
            cmp_vl = st.number_input(
                "V_L [mL]",
                min_value=1.0,
                max_value=250.0,
                value=default_volume,
                step=1.0,
                key=f"cmp_vl_{i}",
            )

        with c2:
            cmp_d0 = st.number_input(
                "d₀ [mm]",
                min_value=5.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                key=f"cmp_d0_{i}",
            )
            cmp_osmo = st.number_input(
                "Osmolality [osmol/kg]",
                min_value=0.01,
                max_value=2.5,
                value=0.30,
                step=0.01,
                key=f"cmp_osmo_{i}",
            )
            cmp_rpm = st.number_input(
                "rpm",
                min_value=50.0,
                max_value=600.0,
                value=250.0,
                step=5.0,
                key=f"cmp_rpm_{i}",
            )

        with c3:
            with st.expander("Advanced gas phase", expanded=False):
                cmp_pressure = st.number_input(
                    "p_R [bar abs.]",
                    min_value=0.1,
                    max_value=5.0,
                    value=STANDARD_PRESSURE_BAR,
                    step=0.1,
                    key=f"cmp_pressure_{i}",
                )
                cmp_o2 = st.number_input(
                    "y_O₂ [-]",
                    min_value=0.01,
                    max_value=1.0,
                    value=STANDARD_O2_FRACTION,
                    step=0.0005,
                    format="%.4f",
                    key=f"cmp_o2_{i}",
                )

        cmp_otr = model.predict_otrmax_mmol_l_h(
            cmp_d, cmp_vl, cmp_d0, cmp_osmo, cmp_pressure, cmp_o2, cmp_rpm
        )
        cmp_status, _, _ = validation_status(cmp_rpm, cmp_vl, cmp_d0, cmp_d)

        rows.append(
            {
                "Setup": setup_name,
                "d [mm]": cmp_d,
                "V_L [mL]": cmp_vl,
                "d0 [mm]": cmp_d0,
                "Osmol [osmol/kg]": cmp_osmo,
                "rpm": cmp_rpm,
                "p_R [bar abs.]": cmp_pressure,
                "y_O2 [-]": cmp_o2,
                "OTRmax [mmol/L/h]": cmp_otr,
                "Expected lower": max(0.0, cmp_otr - EXPECTED_ERROR_MMOL_L_H),
                "Expected upper": cmp_otr + EXPECTED_ERROR_MMOL_L_H,
                "Confidence": cmp_status,
            }
        )

    comparison_df = pd.DataFrame(rows)
    st.markdown("## Comparison result")
    st.dataframe(comparison_df, use_container_width=True)

    fig3, ax3 = plt.subplots()
    ax3.bar(comparison_df["Setup"], comparison_df["OTRmax [mmol/L/h]"])
    ax3.set_ylabel("OTRmax [mmol/L/h]")
    ax3.set_title("Setup comparison")
    ax3.tick_params(axis="x", rotation=30)
    st.pyplot(fig3)

    st.download_button(
        "Download comparison as CSV",
        comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="otrmax_setup_comparison.csv",
        mime="text/csv",
    )
