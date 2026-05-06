import streamlit as st
import pandas as pd
import joblib
import os
import time

MODEL_PATH = "student_exam_pass_fail_model.joblib"

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Student AI Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- THEME SWITCH ----------------
theme = st.toggle("🌙 Dark Mode", value=True)

# ---------------- CSS ----------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.main {{
    background: {"#0e1117" if theme else "#f5f7fa"};
}}

.card {{
    background: {"rgba(255,255,255,0.05)" if theme else "white"};
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
}}

h1, h2, h3, h4 {{
    color: {"white" if theme else "#222"};
}}

.stButton>button {{
    border-radius: 12px;
    height: 50px;
    font-weight: 600;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border: none;
}}

.pass {{
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
}}

.fail {{
    background: linear-gradient(135deg, #cb2d3e, #ef473a);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
}}

.footer {{
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    background: rgba(0,0,0,0.7);
    color: white;
}}

.footer a {{
    color: #00c6ff;
    text-decoration: none;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
MODEL_PATH = r"D:\Universty\Pattern\Project\student_exam_pass_fail_model.joblib"

@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

assets = load_assets()

# ---------------- HEADER ----------------
st.title("🎓 Student Success AI")
st.caption("Predict. Analyze. Improve.")

# ---------------- MAIN ----------------
if assets:

    col1, col2 = st.columns([1,2])

    # -------- INPUT --------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🧾 Student Data")

        hours = st.number_input("Study Hours", 0.1, 24.0, 5.0, 0.5)
        score = st.slider("Previous Score", 0.0, 100.0, 70.0)

        eff_display = score / (hours + 1)
        st.metric("Efficiency", f"{eff_display:.2f}")

        predict_btn = st.button("🚀 Predict Now", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- RESULT --------
    with col2:

        if predict_btn:

            with st.spinner("Analyzing..."):
                time.sleep(1)

                input_df = pd.DataFrame([[hours, score]],
                                        columns=['Study Hours', 'Previous Exam Score'])

                input_df['Efficiency'] = input_df['Previous Exam Score'] / (input_df['Study Hours'] + 1)
                input_df = input_df[['Study Hours', 'Previous Exam Score', 'Efficiency']]

                try:
                    preprocessor = assets['preprocessor']
                    model = assets['model']

                    processed = preprocessor.transform(input_df)
                    prediction = model.predict(processed)[0]

                    probs = model.predict_proba(processed)[0]
                    pass_probability = probs[1] * 100

                    st.subheader("🎯 Result")

                    if prediction == 1:
                        st.markdown(f"""
                        <div class="pass">
                            <h1>PASS ✅</h1>
                            <p>Student is on track for success</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div class="fail">
                            <h1>FAIL ❌</h1>
                            <p>Needs improvement in study habits</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"### 📊 Success Probability: {pass_probability:.2f}%")
                    st.progress(pass_probability / 100)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Result", "PASS" if prediction == 1 else "FAIL")
                    c2.metric("Confidence", f"{pass_probability:.1f}%")
                    c3.metric("Load", "High" if hours > 8 else "Normal")

                except Exception as e:
                    st.error(f"Error: {e}")

        else:
            st.info("👉 Enter data and click Predict")

else:
    st.error("❌ Model not found")

# ---------------- FOOTER ----------------
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    z-index: 999;
}
.footer a {
    color: #4f46e5;
    text-decoration: none;
    font-weight: bold;
}
.footer a:hover {
    text-decoration: underline;
}
</style>

<div class="footer">
    Built by Abdullah 👨‍💻 | 
    <a href="./About" target="_self">About</a>
</div>
""", unsafe_allow_html=True)