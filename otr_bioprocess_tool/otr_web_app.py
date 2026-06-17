import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


class OTRModel:
    def __init__(self):
        # Empirical demonstrator parameters.
        # Replace or calibrate with exact paper-derived / experimental parameters before scientific use.
        self.A = 180
        self.alpha = 1.25
        self.beta = 0.85
        self.gamma = -0.65
        self.delta = -0.4
        self.k_osmo = 0.35

    def kla(self, n_per_s, shaking_diameter_m, filling_volume_m3, flask_diameter_m, osmolality):
        kla_base = (
            self.A
            * (n_per_s ** self.alpha)
            * (shaking_diameter_m ** self.beta)
            * (filling_volume_m3 ** self.gamma)
            * (flask_diameter_m ** self.delta)
        )
        osmo_factor = np.exp(-self.k_osmo * (osmolality - 0.3))
        return kla_base * osmo_factor

    def predict(
        self,
        flask_diameter_mm,
        filling_volume_ml,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar,
        o2_fraction,
        rpm,
    ):
        flask_diameter_m = flask_diameter_mm / 1000
        shaking_diameter_m = shaking_diameter_mm / 1000
        filling_volume_m3 = filling_volume_ml / 1e6
        n_per_s = rpm / 60

        kla = self.kla(
            n_per_s,
            shaking_diameter_m,
            filling_volume_m3,
            flask_diameter_m,
            osmolality_osmol_kg,
        )

        gas_factor = (pressure_bar * o2_fraction) / 0.21
        otrmax = kla * gas_factor
        return otrmax, kla


model = OTRModel()

st.set_page_config(
    page_title="OTR Bioprocess Design Tool",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 OTR Bioprocess Design Tool")
st.caption("Browser-based demonstrator for estimating OTRmax in shaken flask processes.")

with st.expander("Scientific note / limitations", expanded=False):
    st.markdown(
        """
        This app currently uses a practical empirical model structure.

        **Before scientific or customer-facing use**, the coefficients should be
        replaced or calibrated using the exact equation from the referenced paper
        and/or experimental OTRmax data for the target flask/shaker/media system.

        The shaking frequency / RPM is included because it is essential for OTRmax prediction.
        """
    )

st.sidebar.header("Presets")

flask_preset = st.sidebar.selectbox(
    "Flask preset",
    ["Custom", "250 mL Erlenmeyer", "500 mL Erlenmeyer", "1000 mL Erlenmeyer"],
)

preset_values = {
    "Custom": (80.0, 50.0),
    "250 mL Erlenmeyer": (80.0, 50.0),
    "500 mL Erlenmeyer": (105.0, 100.0),
    "1000 mL Erlenmeyer": (130.0, 200.0),
}

default_diameter, default_volume = preset_values[flask_preset]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input parameters")

    flask_diameter_mm = st.number_input(
        "Shake flask diameter [mm]",
        min_value=10.0,
        max_value=250.0,
        value=default_diameter,
        step=1.0,
    )

    filling_volume_ml = st.number_input(
        "Filling volume [mL]",
        min_value=1.0,
        max_value=1000.0,
        value=default_volume,
        step=1.0,
    )

    shaking_diameter_mm = st.number_input(
        "Shaking diameter [mm]",
        min_value=1.0,
        max_value=100.0,
        value=25.0,
        step=1.0,
    )

    osmolality_osmol_kg = st.number_input(
        "Media osmolality [osmol/kg]",
        min_value=0.05,
        max_value=3.0,
        value=0.3,
        step=0.05,
    )

    pressure_bar = st.number_input(
        "Pressure [bar]",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

    o2_fraction = st.number_input(
        "O₂ concentration / fraction [-]",
        min_value=0.0,
        max_value=1.0,
        value=0.21,
        step=0.01,
    )

    rpm = st.number_input(
        "Shaking frequency [rpm]",
        min_value=50.0,
        max_value=500.0,
        value=250.0,
        step=5.0,
    )

    our = st.number_input(
        "Optional OUR [mmol/L/h]",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=5.0,
    )

with col2:
    st.subheader("Calculated result")

    otrmax, kla = model.predict(
        flask_diameter_mm,
        filling_volume_ml,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar,
        o2_fraction,
        rpm,
    )

    st.metric("Predicted OTRmax", f"{otrmax:.2f} mmol/L/h")
    st.metric("Estimated kLa", f"{kla:.2f} 1/h")

    if our > otrmax:
        st.error("⚠️ Oxygen limitation likely: OUR is higher than predicted OTRmax.")
    else:
        st.success("✅ Oxygen supply appears sufficient for the provided OUR.")

    if rpm > 350:
        st.warning("High shaking speed: prediction may be outside common empirical range.")
    if osmolality_osmol_kg > 1.0:
        st.warning("High osmolality: oxygen transfer and solubility effects may need stronger correction.")

st.subheader("OTRmax vs shaking frequency")

rpm_range = np.linspace(50, 500, 100)
otr_values = [
    model.predict(
        flask_diameter_mm,
        filling_volume_ml,
        shaking_diameter_mm,
        osmolality_osmol_kg,
        pressure_bar,
        o2_fraction,
        r,
    )[0]
    for r in rpm_range
]

fig, ax = plt.subplots()
ax.plot(rpm_range, otr_values)
ax.axvline(rpm, linestyle="--")
ax.set_xlabel("Shaking frequency [rpm]")
ax.set_ylabel("Predicted OTRmax [mmol/L/h]")
ax.set_title("OTRmax vs shaking frequency")
st.pyplot(fig)

st.subheader("Export current calculation")

export_df = pd.DataFrame(
    [
        {
            "shake_flask_diameter_mm": flask_diameter_mm,
            "filling_volume_ml": filling_volume_ml,
            "shaking_diameter_mm": shaking_diameter_mm,
            "media_osmolality_osmol_kg": osmolality_osmol_kg,
            "pressure_bar": pressure_bar,
            "o2_fraction": o2_fraction,
            "rpm": rpm,
            "our_mmol_L_h": our,
            "predicted_otrmax_mmol_L_h": otrmax,
            "estimated_kla_1_h": kla,
        }
    ]
)

st.dataframe(export_df, use_container_width=True)

st.download_button(
    "Download result as CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name="otrmax_result.csv",
    mime="text/csv",
)
