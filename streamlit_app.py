import streamlit as st
import sys
import json
from io import StringIO
import matplotlib.pyplot as plt

# Import your simulation module
# Assuming your code is saved as crop_simulation.py
from crop_simulation import (
    get_ml_features,
    simulate_crop_growth_realistic,
    visualize_results,
    print_comprehensive_summary,
    export_result_to_json_realistic,
    run_sensitivity_analysis,
    CROP_PARAMS
)

# Page config
st.set_page_config(
    page_title="Crop Growth Simulation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background-color: #251;
        color:#f8f9fa;
        border-radius:20px;
        padding-block:15px;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .sub-header {
        text-align: center;
        color: #34495e;
        font-size: 1.2em;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #251;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #251;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌾 Crop Growth Simulation System - Egypt</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Realistic crop growth simulation using real-world data</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    
    # Location input
    st.subheader("📍 Location")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=30.646027, format="%.6f")
    with col2:
        longitude = st.number_input("Longitude", value=31.149082, format="%.6f")
    
    # Crop selection
    st.subheader("🌱 Select Crop")
    crop_name = st.selectbox(
        "Crop Type",
        options=list(CROP_PARAMS.keys()),
        index=0
    )
    
    # Show crop info
    if crop_name:
        crop_info = CROP_PARAMS[crop_name]
        st.info(f"""
        **Crop Information:**
        - Season: {crop_info['planting_season']}
        - Duration: {crop_info['season_days']} days
        - Optimal pH: {crop_info['optimal_ph'][0]}-{crop_info['optimal_ph'][1]}
        - Optimal Temperature: {crop_info['optimal_temp'][0]}-{crop_info['optimal_temp'][1]}°C
        """)
    
    # Analysis options
    st.subheader("📊 Analysis Options")
    run_sensitivity = st.checkbox("Run Sensitivity Analysis", value=False)
    show_visualization = st.checkbox("Show Visualizations", value=True)
    
    # Run button
    run_simulation = st.button("🚀 Run Simulation", type="primary")

# Main content
if run_simulation:
    with st.spinner("Fetching data from API..."):
        try:
            # Get API data
            api_features = get_ml_features(latitude, longitude)
            st.success("✅ Data fetched successfully!")
            
            # Show API data
            with st.expander("📋 View API Data"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature", f"{api_features.get('temperature', 0):.1f}°C")
                    st.metric("Humidity", f"{api_features.get('humidity', 0):.0f}%")
                with col2:
                    st.metric("NDVI", f"{api_features.get('ndvi', 0):.3f}")
                    st.metric("pH", f"{api_features.get('ph', 0)/10:.1f}")
                with col3:
                    st.metric("Sand", f"{api_features.get('sand', 0):.0f} g/kg")
                    st.metric("Clay", f"{api_features.get('clay', 0):.0f} g/kg")
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.stop()
    
    with st.spinner("Running simulation..."):
        try:
            # Run simulation
            result = simulate_crop_growth_realistic(api_features, crop_name)
            
            if result["success"]:
                st.success("✅ Simulation completed successfully!")
                
                # Display key metrics
                st.header("📊 Key Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Actual Yield",
                        f"{result['predicted_yield']:.2f} ton/ha",
                        delta=f"{result['yield_efficiency']:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Potential Yield",
                        f"{result['potential_yield']:.2f} ton/ha"
                    )
                
                with col3:
                    st.metric(
                        "Total Irrigation",
                        f"{result['total_irrigation']:.1f} mm"
                    )
                
                with col4:
                    st.metric(
                        "Total Rainfall",
                        f"{result['total_rainfall']:.1f} mm"
                    )
                
                # Stress factors
                st.header("⚠️ Stress Factors")
                col1, col2, col3 = st.columns(3)
                
                stress = result['stress_factors']
                with col1:
                    st.metric("Soil Quality", f"{stress['soil']:.1%}")
                with col2:
                    st.metric("Water Adequacy", f"{stress['water']:.1%}")
                with col3:
                    st.metric("Temperature Suitability", f"{stress['temperature']:.1%}")
                
                # Fertilizer recommendations
                st.header("💊 Fertilizer Recommendations")
                fert = result['fertilizer_needs']
                
                if any(v > 10 for v in fert.values()):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if fert['nitrogen'] > 10:
                            st.warning(f"🔵 Nitrogen: {fert['nitrogen']:.1f} kg/ha")
                    with col2:
                        if fert['phosphorus'] > 10:
                            st.warning(f"🟠 Phosphorus: {fert['phosphorus']:.1f} kg/ha")
                    with col3:
                        if fert['potassium'] > 10:
                            st.warning(f"🟢 Potassium: {fert['potassium']:.1f} kg/ha")
                else:
                    st.success("✅ Soil nutrients are adequate")
                
                # Visualization
                if show_visualization:
                    st.header("📈 Charts & Graphs")
                    
                    # Create tabs for different plots
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Biomass Growth", 
                        "Water Balance", 
                        "Stress Factors",
                        "Growth Stages"
                    ])
                    
                    with tab1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(result["days"], result["biomass"], linewidth=2.5, color='#2ecc71')
                        ax.set_title('Biomass Growth', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Day')
                        ax.set_ylabel('Biomass (ton/ha)')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with tab2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(result["days"], result["swc_content"], linewidth=2, color='#004d40')
                        fc = CROP_PARAMS[crop_name]['fc']
                        pwp = CROP_PARAMS[crop_name]['pwp']
                        ax.axhline(y=fc, color='#3498db', linestyle='--', label='FC')
                        ax.axhline(y=pwp, color='#e74c3c', linestyle='--', label='PWP')
                        ax.set_title('Soil Water Content', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Day')
                        ax.set_ylabel('SWC (m³/m³)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with tab3:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(result["days"], result["water_stress"], label='Water Stress', color='#3498db')
                        ax.plot(result["days"], result["temperature_stress"], label='Temperature Stress', color='#e74c3c')
                        ax.set_title('Daily Stress Factors', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Day')
                        ax.set_ylabel('Stress Factor (0-1)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with tab4:
                        stages_summary = result['stage_summary']
                        stage_names = list(stages_summary.keys())
                        stage_days = [stages_summary[s]['days'] for s in stage_names]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(stage_names, stage_days, color='#251', alpha=0.7)
                        ax.set_title('Growth Stage Duration', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Stage')
                        ax.set_ylabel('Days')
                        ax.grid(True, alpha=0.3, axis='y')
                        st.pyplot(fig)
                
                # Sensitivity Analysis
                if run_sensitivity:
                    st.header("🔬 Sensitivity Analysis")
                    with st.spinner("Running sensitivity analysis..."):
                        sensitivity_json, scenarios = run_sensitivity_analysis(
                            api_features, crop_name, result
                        )
                        
                        st.success("✅ Sensitivity analysis completed!")
                        
                        # Parse and display sensitivity results
                        sensitivity_data = json.loads(sensitivity_json)
                        
                        # Best and worst scenarios
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"**Best Scenario:** {sensitivity_data['summary_metrics']['best_scenario']}")
                            st.metric("Highest Yield", f"{sensitivity_data['summary_metrics']['best_yield']:.2f} ton/ha")
                        
                        with col2:
                            st.error(f"**Worst Scenario:** {sensitivity_data['summary_metrics']['worst_scenario']}")
                            st.metric("Lowest Yield", f"{sensitivity_data['summary_metrics']['worst_yield']:.2f} ton/ha")
                        
                        # Recommendations
                        if 'recommendations' in sensitivity_data:
                            st.subheader("💡 Recommendations")
                            for rec in sensitivity_data['recommendations']:
                                if rec['impact'] == 'positive':
                                    st.success(f"✅ {rec['message']}")
                                elif rec['impact'] == 'warning':
                                    st.warning(f"⚠️ {rec['message']}")
                                elif rec['impact'] == 'critical':
                                    st.error(f"❌ {rec['message']}")
                                else:
                                    st.info(f"ℹ️ {rec['message']}")
                                
                                st.caption(f"Suggested Action: {rec['suggested_action']}")
                
                # Export options
                st.header("💾 Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export JSON
                    json_data = export_result_to_json_realistic(result, crop_name)
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json_data,
                        file_name=f"{crop_name}_simulation_{latitude}_{longitude}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Export sensitivity analysis if run
                    if run_sensitivity:
                        st.download_button(
                            label="📥 Download Sensitivity Analysis (JSON)",
                            data=sensitivity_json,
                            file_name=f"{crop_name}_sensitivity_{latitude}_{longitude}.json",
                            mime="application/json"
                        )
                
            else:
                st.error("❌ Simulation failed")
                
        except Exception as e:
            st.error(f"❌ Simulation error: {str(e)}")
            st.exception(e)

else:
    # Welcome screen
    st.info("""
    ### 👋 Welcome to the Crop Growth Simulation System
    
    This system uses:
    - ✅ Real data from soil and weather APIs
    - ✅ Accurate physical models (Penman-Monteith, Farquhar)
    - ✅ Calibrated for Egyptian conditions
    - ✅ Comprehensive stress factor analysis
    
    **To get started:**
    1. Select location (latitude and longitude)
    2. Choose crop type
    3. Click "Run Simulation"
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>Crop Growth Simulation System v2.0 | Designed for Egyptian Conditions</p>
</div>
""", unsafe_allow_html=True)