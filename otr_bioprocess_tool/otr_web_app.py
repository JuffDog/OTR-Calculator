import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


class MeierOTRModel:
    """
    Paper-based OTRmax correlation for non-baffled shake flasks.

    OTRmax = 3.72e-4
             * Osmol^0.05
             * n^(1.18 - Osmol/10.1)
             * V_L^-0.74
             * d0^0.33
             * d^1.88
             * p_R
             * y_O2

    Units:
    - OTRmax: mmol/L/h
    - Osmol: osmol/kg
    - n: rpm
    - V_L: mL
    - d0: cm
    - d: mm
    - p_R: bar absolute
    - y_O2: O2 mole fraction
    """

    def predict_otrmax(
        self,
        flask_diameter_mm,
        filling_volume_ml,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar=1.0,
        o2_fraction=0.2095,
        rpm=250,
    ):
        d_mm = flask_diameter_mm
        vl_ml = filling_volume_ml
        d0_cm = shaking_diameter_mm / 10.0
        osmol = osmolality_osmol_kg
        n = rpm
        p_r = pressure_bar
        y_o2 = o2_fraction

        otrmax = (
            3.72e-4
            * (osmol ** 0.05)
            * (n ** (1.18 - osmol / 10.1))
            * (vl_ml ** -0.74)
            * (d0_cm ** 0.33)
            * (d_mm ** 1.88)
            * p_r
            * y_o2
        )

        return otrmax

    def critical_rpm_for_our(
        self,
        our_mmol_l_h,
        flask_diameter_mm,
        filling_volume_ml,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar=1.0,
        o2_fraction=0.2095,
    ):
        if our_mmol_l_h <= 0:
            return None

        d_mm = flask_diameter_mm
        vl_ml = filling_volume_ml
        d0_cm = shaking_diameter_mm / 10.0
        osmol = osmolality_osmol_kg
        p_r = pressure_bar
        y_o2 = o2_fraction

        exponent = 1.18 - osmol / 10.1

        prefactor = (
            3.72e-4
            * (osmol ** 0.05)
            * (vl_ml ** -0.74)
            * (d0_cm ** 0.33)
            * (d_mm ** 1.88)
            * p_r
            * y_o2
        )

        if prefactor <= 0 or exponent <= 0:
            return None

        return (our_mmol_l_h / prefactor) ** (1 / exponent)


model = MeierOTRModel()

STANDARD_PRESSURE_BAR = 1.0
STANDARD_O2_FRACTION = 0.2095

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
        The app uses the empirical OTRmax correlation:

        $$
        OTR_{max} =
        3.72 \times 10^{-4}
        \cdot Osmol^{0.05}
        \cdot n^{(1.18 - Osmol/10.1)}
        \cdot V_L^{-0.74}
        \cdot d_0^{0.33}
        \cdot d^{1.88}
        \cdot p_R
        \cdot y_{O_2}
        $$

        **Required units in the equation**

        - $OTR_{max}$: mmol L⁻¹ h⁻¹
        - $Osmol$: osmol kg⁻¹
        - $n$: rpm / min⁻¹
        - $V_L$: mL
        - $d_0$: cm — the app converts your mm input to cm
        - $d$: mm
        - $p_R$: bar absolute
        - $y_{O_2}$: O₂ mole fraction, e.g. 0.2095 for air

        This correlation is intended for **non-baffled shake flasks** within the validated experimental range.
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

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input parameters")

    flask_diameter_mm = st.number_input(
        "Shake flask maximum diameter d [mm]",
        min_value=30.0,
        max_value=200.0,
        value=default_diameter,
        step=1.0,
    )

    filling_volume_ml = st.number_input(
        "Filling volume V_L [mL]",
        min_value=1.0,
        max_value=250.0,
        value=default_volume,
        step=1.0,
    )

    shaking_diameter_mm = st.number_input(
        "Shaking diameter d₀ [mm]",
        min_value=5.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
    )

    osmolality_osmol_kg = st.number_input(
        "Media osmolality [osmol/kg]",
        min_value=0.01,
        max_value=2.50,
        value=0.30,
        step=0.01,
    )

    rpm = st.number_input(
        "Shaking frequency n [rpm]",
        min_value=50.0,
        max_value=600.0,
        value=250.0,
        step=5.0,
    )

    our = st.number_input(
        "Optional OUR [mmol/L/h]",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=1.0,
    )

    with st.expander("Advanced settings: gas phase", expanded=False):
        st.caption("Default corresponds approximately to ambient air at 1 bar absolute.")
        pressure_bar = st.number_input(
            "Pressure p_R [bar abs.]",
            min_value=0.1,
            max_value=5.0,
            value=STANDARD_PRESSURE_BAR,
            step=0.1,
        )

        o2_fraction = st.number_input(
            "O₂ mole fraction y_O₂ [-]",
            min_value=0.01,
            max_value=1.0,
            value=STANDARD_O2_FRACTION,
            step=0.0005,
            format="%.4f",
        )

        reset_gas = st.button("Reset gas phase to ambient standard")
        if reset_gas:
            pressure_bar = STANDARD_PRESSURE_BAR
            o2_fraction = STANDARD_O2_FRACTION
            st.info("For reset, reload the app or manually set pressure to 1.0 and O₂ to 0.2095.")

with col2:
    st.subheader("Calculated result")
    st.caption(f"Gas phase used: p_R = {pressure_bar:.3f} bar abs., y_O₂ = {o2_fraction:.4f}")

    otrmax = model.predict_otrmax(
        flask_diameter_mm,
        filling_volume_ml,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar,
        o2_fraction,
        rpm,
    )

    st.metric("Predicted OTRmax", f"{otrmax:.2f} mmol/L/h")

    if our > 0:
        if our > otrmax:
            st.error("⚠️ Oxygen limitation likely: OUR is higher than predicted OTRmax.")
        else:
            st.success("✅ Oxygen supply appears sufficient for the provided OUR.")

        crit_rpm = model.critical_rpm_for_our(
            our,
            flask_diameter_mm,
            filling_volume_ml,
            shaking_diameter_mm,
            osmolality_osmol_kg,
            pressure_bar,
            o2_fraction,
        )

        if crit_rpm is not None:
            st.metric("Approx. critical rpm for provided OUR", f"{crit_rpm:.0f} rpm")

    st.markdown("### Plausibility checks")

    warnings = []

    if not (100 <= rpm <= 450):
        warnings.append("RPM is outside the paper's broad validation range of approximately 100–450 rpm.")
    if not (2 <= filling_volume_ml <= 160):
        warnings.append("Filling volume is outside the reported validation range of approximately 2–160 mL.")
    if not (12.5 <= shaking_diameter_mm <= 100):
        warnings.append("Shaking diameter is outside the reported validation range of approximately 12.5–100 mm.")
    if not (51 <= flask_diameter_mm <= 131):
        warnings.append("Flask diameter is outside the reported validation range of approximately 51–131 mm.")

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("Inputs are within the broad reported validation ranges.")

st.subheader("OTRmax vs shaking frequency")

rpm_range = np.linspace(50, 600, 150)
otr_values = [
    model.predict_otrmax(
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

if our > 0:
    ax.axhline(our, linestyle=":", label="provided OUR")

ax.set_xlabel("Shaking frequency n [rpm]")
ax.set_ylabel("Predicted OTRmax [mmol/L/h]")
ax.set_title("OTRmax vs shaking frequency")
ax.legend()
st.pyplot(fig)

st.subheader("Design-space heatmap")

heatmap_enabled = st.checkbox("Show heatmap: OTRmax over rpm × filling volume", value=False)

if heatmap_enabled:
    rpm_min, rpm_max = st.slider("RPM range", 50, 600, (100, 450), step=10)
    vl_min, vl_max = st.slider("Filling volume range [mL]", 1, 250, (10, 100), step=1)

    rpm_grid = np.linspace(rpm_min, rpm_max, 80)
    vl_grid = np.linspace(vl_min, vl_max, 80)
    R, V = np.meshgrid(rpm_grid, vl_grid)

    Z = model.predict_otrmax(
        flask_diameter_mm,
        V,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar,
        o2_fraction,
        R,
    )

    fig2, ax2 = plt.subplots()
    contour = ax2.contourf(R, V, Z, levels=20)
    fig2.colorbar(contour, ax=ax2, label="OTRmax [mmol/L/h]")
    ax2.set_xlabel("Shaking frequency n [rpm]")
    ax2.set_ylabel("Filling volume V_L [mL]")
    ax2.set_title("Design-space heatmap")
    st.pyplot(fig2)

st.subheader("Export current calculation")

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
            "provided_our_mmol_L_h": our,
            "predicted_otrmax_mmol_L_h": otrmax,
        }
    ]
)

st.dataframe(export_df, use_container_width=True)

st.download_button(
    "Download result as CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name="otrmax_paper_model_result.csv",
    mime="text/csv",
)
