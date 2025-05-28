# app.py
import streamlit as st

st.title("🌐 Geodesic Dome Analysis App")

st.sidebar.header("📐 Dome Parameters")
dome_type = st.sidebar.selectbox("Dome Type", options=["Tetrahedron", "Octahedron", "Icosahedron"])
type_map = {"Tetrahedron": 1, "Octahedron": 2, "Icosahedron": 3}
span = st.sidebar.number_input("Span [mm]", value=500)
height = st.sidebar.number_input("Height [mm]", value=250)
freq = st.sidebar.number_input("Frequency (division)", value=2)

st.sidebar.header("🔩 Material Properties")
E = st.sidebar.number_input("Elastic Modulus [MPa]", value=200000)
A = st.sidebar.number_input("Cross Section Area [mm²]", value=100)
rho = st.sidebar.number_input("Density [kg/m³]", value=7850)

st.sidebar.header("⚙️ Load Options")
apply_deadload = st.sidebar.checkbox("Apply Dead Load", value=True)
ext_horizontal = st.sidebar.number_input("Horizontal Load [N]", value=0)
ext_vertical = st.sidebar.number_input("Vertical Load [N]", value=0)
pointload = st.sidebar.text_input("Point Load (Format: node_id, Fx, Fy, Fz)", "1, 0, 0, -500")

if st.button("🧠 Run Analysis"):
    try:
        pointload_parsed = [list(map(float, p.strip().split(","))) for p in pointload.split(";")]
    except:
        st.error("Point load format is incorrect.")
        pointload_parsed = []

    dome_input = {
        "type": type_map[dome_type],
        "span": span,
        "height": height,
        "freq": freq,
        "geomat": [A, E, rho],
        "apply_deadload": apply_deadload,
        "ext_horizontal": ext_horizontal,
        "ext_vertical": ext_vertical,
        "pointload": pointload_parsed
    }

    with st.spinner("Running structural and dynamic analysis..."):
        dome = run_geodesic_analysis(dome_input)
        st.success("Analysis completed ✅")

        st.subheader("📊 Basic Info")
        st.json(dome["info"])

        st.subheader("📈 Dynamic Properties")
        st.write("Frequencies (Hz):", dome["dynamic"]["frequencies"])
        st.write("Periods (s):", dome["dynamic"]["periods"])

        st.subheader("🖼️ Visualize Dome")
        img_buffer = plot_dome(dome, scale=0, show_labels=True, label_type="group")
        st.image(img_buffer, caption="Geodesic Dome", use_container_width=True)

        st.download_button("📥 Download Dome Data", data=str(dome), file_name="dome_data.txt")
        
        st.plotly_chart(plot_dome_plotly(dome, scale=0, label_type="group"), use_container_width=True)

