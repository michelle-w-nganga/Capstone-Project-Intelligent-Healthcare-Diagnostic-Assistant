# ============================================================  
# CAPSTONE MAIN APPLICATION  
# Intelligent Healthcare Diagnostic Assistant  
# Introduction to AI — 13-Week Capstone  
# ============================================================  

import os
import sys  
import json  
import warnings  
import argparse
import subprocess
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
warnings.filterwarnings('ignore')  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Streamlit if available for Interactive Web UI
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Import all sub-modules  
from modules.agent           import HealthcareDiagnosticAgent, PatientPercept  
from modules.knowledge_base  import MedicalKnowledgeBase  
from modules.bayesian_net    import SimpleBayesianDiagnostics  
from modules.ml_classifier   import MLDiagnosticClassifier  
from modules.neural_network  import NeuralDiagnosticModel  
from modules.fuzzy_controller import FuzzySeverityAssessor  
from modules.planner         import TreatmentPlanner  

# ── ANSI Colors ────────────────────────────────────────────  
class C:  
    HEADER = '\033[95m'; BLUE   = '\033[94m'  
    GREEN  = '\033[92m'; YELLOW = '\033[93m'  
    RED    = '\033[91m'; BOLD   = '\033[1m'  
    END    = '\033[0m'  

def banner():  
    print(f"""  
{C.BOLD}{C.BLUE}  
╔══════════════════════════════════════════════════════════╗  
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║  
║         Introduction to AI — Capstone Project            ║  
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║  
╚══════════════════════════════════════════════════════════╝  
{C.END}""")  

def section(title: str):  
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  

def build_system() -> HealthcareDiagnosticAgent:  
    """Instantiate and wire all AI modules"""  
    agent = HealthcareDiagnosticAgent()  
    
    # 1. Logic Knowledge Base
    kb = MedicalKnowledgeBase()
    agent.register_module('KnowledgeBase', kb)

    # 2. Bayesian Network
    bn = SimpleBayesianDiagnostics()
    agent.register_module('BayesianNet', bn)

    # 3. ML Classifier
    ml = MLDiagnosticClassifier()
    ml.train(verbose=False)
    agent.register_module('MLClassifier', ml)

    # 4. Neural Network
    nn = NeuralDiagnosticModel()
    nn.train(epochs=20, verbose=0)
    agent.register_module('NeuralNetwork', nn)

    # 5. Fuzzy Logic Controller
    fuzzy = FuzzySeverityAssessor()
    agent.register_module('FuzzySeverity', fuzzy)

    # 6. AI Treatment Planner
    planner = TreatmentPlanner()
    agent.register_module('Planner', planner)

    return agent

def get_sample_patients():
    """Returns standard test cases covering various diseases and clinical severities"""
    return [
        PatientPercept(
            patient_id="P001",
            symptoms=["fever", "cough", "loss of smell", "fatigue"],
            age=34,
            temperature=38.9,
            heart_rate=98,
            blood_pressure="120/80"
        ),
        PatientPercept(
            patient_id="P002",
            symptoms=["chest pain", "shortness of breath", "sweating"],
            age=62,
            temperature=37.2,
            heart_rate=125,
            blood_pressure="150/95"
        ),
        PatientPercept(
            patient_id="P003",
            symptoms=["headache", "stiff neck", "fever", "light sensitivity"],
            age=22,
            temperature=39.8,
            heart_rate=110,
            blood_pressure="115/75"
        ),
        PatientPercept(
            patient_id="P004",
            symptoms=["frequent urination", "excessive thirst", "fatigue", "blurred vision"],
            age=45,
            temperature=36.8,
            heart_rate=76,
            blood_pressure="130/85"
        ),
        PatientPercept(
            patient_id="P005",
            symptoms=["fever", "rash", "joint pain", "headache", "body aches"],
            age=29,
            temperature=38.6,
            heart_rate=92,
            blood_pressure="110/70"
        ),
    ]

def display_report(patient: PatientPercept, report: dict, planner: TreatmentPlanner):
    """Outputs structured formatting for patient assessment"""
    print(f"\n{C.BOLD}{C.BLUE}📋 DIAGNOSTIC REPORT: Patient {patient.patient_id}{C.END}")
    print("─" * 60)
    print(f"  Age:             {patient.age}")
    print(f"  Vitals:          Temp {patient.temperature}°C | HR {patient.heart_rate} BPM | BP {patient.blood_pressure}")
    print(f"  Symptoms:        {', '.join(patient.symptoms)}")
    print("─" * 60)
    print(f"  {C.BOLD}Primary Diagnosis:{C.END} {C.GREEN}{report['diagnosis']}{C.END}")
    print(f"  {C.BOLD}Mean Confidence:  {C.END} {report['confidence']:.2%}")
    
    urgency_color = C.RED if report['urgency'] in ['CRITICAL', 'HIGH'] else C.YELLOW
    print(f"  {C.BOLD}Urgency Level:    {C.END} {urgency_color}{report['urgency']}{C.END}")
    print(f"  {C.BOLD}System Action:    {C.END} {report['next_action']}")
    
    print(f"\n  {C.BOLD}Sub-Module Diagnoses Breakdown:{C.END}")
    for mod_name, res in report['module_results'].items():
        if isinstance(res, dict) and 'summary' in res:
            print(f"    • {mod_name:<16}: {res['summary']}")

            # ✅ NEW NORMALIZED LINE
        diag_key = report['diagnosis'].lower().strip()

        # Map common disease variations to standard keys expected by planner
        if "covid" in diag_key:
            diag_key = "covid-19"
        elif "flu" in diag_key or "influenza" in diag_key:
            diag_key = "flu"

        tx_plan = planner.create_treatment_plan(diag_key, report['urgency'])

def plot_capstone_dashboard(all_reports: list):
    """Plots combined evaluation metrics and exports final dashboard image"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    patients = [r['patient_id'] for r in all_reports]
    confidences = [r['confidence'] * 100 for r in all_reports]
    diagnoses = [r['diagnosis'] for r in all_reports]
    
    bars = ax1.bar(patients, confidences, color='#3498db', edgecolor='black')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Confidence Score (%)", fontweight='bold')
    ax1.set_title("Patient Diagnosis Confidence Scores", fontweight='bold')
    
    for bar, diag in zip(bars, diagnoses):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval / 2, diag, ha='center', va='center', color='white', fontweight='bold', rotation=90)

    ax2 = fig.add_subplot(gs[0, 1])
    urgencies = [r['urgency'] for r in all_reports]
    u_counts = {u: urgencies.count(u) for u in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']}
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    
    ax2.pie([v for v in u_counts.values() if v > 0], 
            labels=[k for k, v in u_counts.items() if v > 0], 
            autopct='%1.1f%%', colors=[c for k, c in zip(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], colors) if u_counts[k] > 0],
            startangle=140)
    ax2.set_title("Patient Urgency Triage Distribution", fontweight='bold')

    ax3 = fig.add_subplot(gs[1, :])
    modules = ['KnowledgeBase', 'BayesianNet', 'MLClassifier', 'NeuralNetwork', 'FuzzySeverity']
    matrix = []

    for r in all_reports:
        row = []
        for m in modules:
            res = r['module_results'].get(m, {})
            row.append(res.get('confidence', 0.5) if isinstance(res, dict) else 0.0)
        matrix.append(row)

    im = ax3.imshow(np.array(matrix).T, cmap='YlGnBu', vmin=0, vmax=1)
    ax3.set_xticks(range(len(patients)))
    ax3.set_xticklabels(patients)
    ax3.set_yticks(range(len(modules)))
    ax3.set_yticklabels(modules)
    ax3.set_title("Sub-Module Confidence Matrix per Patient Case", fontweight='bold')
    fig.colorbar(im, ax=ax3, orientation='vertical', label='Confidence')

    plt.suptitle("Intelligent Healthcare Diagnostic Assistant — Capstone Summary", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("final_capstone_dashboard.png", dpi=150, bbox_inches='tight')
    print(f"\n{C.GREEN}✅ Dashboard graphic saved: final_capstone_dashboard.png{C.END}")

# ── STREAMLIT INTERACTIVE WEB DASHBOARD ────────────────────
def run_interactive_web_ui():
    """Renders interactive Streamlit interface when run via Streamlit"""
    st.set_page_config(page_title="Healthcare Diagnostic AI", layout="wide", page_icon="🏥")
    
    st.title("🏥 Intelligent Healthcare Diagnostic Assistant")
    st.markdown("### Interactive Patient Assessment & Clinical Triage")

    @st.cache_resource
    def get_cached_system():
        agent = build_system()
        planner = agent._modules['Planner']
        return agent, planner

    with st.spinner("Initializing AI Sub-Modules..."):
        agent, planner = get_cached_system()

    st.sidebar.header("⚙️ Input Vitals & Symptoms")
    
    # 1. Patient Metadata & Vitals Inputs
    patient_id = st.sidebar.text_input("Patient ID", "P-CUSTOM")
    age = st.sidebar.number_input("Age (Years)", min_value=1, max_value=110, value=34, step=1)
    
    # Temperature Input (Slider + Exact Number Box)
    temperature = st.sidebar.number_input(
        "Body Temperature (°C)", 
        min_value=34.0, 
        max_value=42.0, 
        value=38.9, 
        step=0.1,
        format="%.1f"
    )
    
    # Heart Rate Input
    heart_rate = st.sidebar.number_input(
        "Heart Rate (BPM)", 
        min_value=30, 
        max_value=220, 
        value=98, 
        step=1
    )
    
    # Blood Pressure Input
    blood_pressure = st.sidebar.text_input("Blood Pressure (Systolic/Diastolic)", "120/80")

    # 2. Symptom Selection
    st.sidebar.subheader("🤒 Symptoms Checklist")
    all_symptoms = [
        "fever", "cough", "loss of smell", "fatigue", "chest pain",
        "shortness of breath", "sweating", "headache", "stiff neck",
        "light sensitivity", "frequent urination", "excessive thirst",
        "blurred vision", "rash", "joint pain", "body aches"
    ]
    
    selected_symptoms = st.sidebar.multiselect(
        "Select Observed Symptoms", 
        all_symptoms, 
        default=["fever", "cough", "loss of smell", "fatigue"]
    )
    
    custom_symptom_input = st.sidebar.text_input("Additional Symptoms (comma-separated)", "")
    if custom_symptom_input.strip():
        extra_symptoms = [s.strip().lower() for s in custom_symptom_input.split(",") if s.strip()]
        selected_symptoms.extend(extra_symptoms)

    # 3. Main Assessment Output
    st.markdown("---")
    col_input, col_action = st.columns([1, 1])
    
    with col_input:
        st.subheader("📋 Current Patient Profile")
        st.write(f"• **Patient ID:** `{patient_id}` | **Age:** {age}")
        st.write(f"• **Temperature:** `{temperature}°C` ({'Fever Detected' if temperature >= 38.0 else 'Normal'})")
        st.write(f"• **Heart Rate:** `{heart_rate} BPM` ({'Tachycardia' if heart_rate > 100 else 'Normal'})")
        st.write(f"• **Blood Pressure:** `{blood_pressure}`")
        st.write(f"• **Symptoms List:** {', '.join(selected_symptoms) if selected_symptoms else 'None specified'}")

    run_btn = st.sidebar.button("⚡ Run Diagnostic Engine", type="primary")

    if run_btn or True:
        patient = PatientPercept(
            patient_id=patient_id,
            symptoms=selected_symptoms,
            age=age,
            temperature=temperature,
            heart_rate=heart_rate,
            blood_pressure=blood_pressure
        )
        
        report = agent.run(patient)
        tx_plan = planner.create_treatment_plan(report['diagnosis'], report['urgency'])

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.subheader("📊 Diagnostic Consensus")
            st.success(f"**Primary Diagnosis:** {report['diagnosis'].upper()}")
            st.metric("Mean Consensus Confidence", f"{report['confidence']:.1%}")
            
            urg_color = "🔴" if report['urgency'] in ["HIGH", "CRITICAL"] else "🟡"
            st.markdown(f"**Fuzzy Urgency Triage:** {urg_color} `{report['urgency']}`")
            st.info(f"**Recommended Action:** {report['next_action']}")

            st.write("---")
            st.subheader("🧩 Sub-Module Diagnostic Breakdown")
            for mod, data in report['module_results'].items():
                if isinstance(data, dict) and 'summary' in data:
                    st.write(f"• **{mod}**: {data['summary']}")

        with res_col2:
            st.subheader("🛠️ STRIPS Action Plan")
            st.write(f"**Total Execution Steps:** {tx_plan.get('steps', 0)}")
            for step in tx_plan.get('plan', []):
                st.markdown(f"**Step {step['step']}:** `{step['action']}` — *[{step['duration']}]*")

# ── INTERACTIVE TERMINAL INPUT MODE ───────────────────────
def run_interactive_cli(agent, planner):
    """Collects custom vitals and symptoms directly via terminal prompts"""
    section("🎮 Custom Patient Input Mode")
    
    try:
        patient_id = input("Enter Patient ID [e.g. P100]: ").strip() or "P100"
        age = int(input("Enter Age (1-100) [Default: 35]: ").strip() or "35")
        temp = float(input("Enter Body Temperature in °C [Default: 38.9]: ").strip() or "38.9")
        hr = int(input("Enter Heart Rate in BPM [Default: 98]: ").strip() or "98")
        bp = input("Enter Blood Pressure [Default: 120/80]: ").strip() or "120/80"
        
        print("\nAvailable preset symptoms: fever, cough, loss of smell, fatigue, chest pain, shortness of breath, sweating, headache, stiff neck, rash, joint pain")
        symptoms_str = input("Enter comma-separated symptoms: ").strip().lower()
        symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()] or ["fever", "cough"]

        patient = PatientPercept(
            patient_id=patient_id,
            symptoms=symptoms,
            age=age,
            temperature=temp,
            heart_rate=hr,
            blood_pressure=bp
        )

        report = agent.run(patient)
        display_report(patient, report, planner)
    except (EOFError, KeyboardInterrupt):
        print("\nReturning to default test suite execution...")

def main():
    if HAS_STREAMLIT and st.runtime.exists():
        run_interactive_web_ui()
        return

    parser = argparse.ArgumentParser(description="Healthcare Diagnostic Assistant")
    parser.add_argument("--gui", action="store_true", help="Launch interactive web UI")
    args, _ = parser.parse_known_args()

    if args.gui and HAS_STREAMLIT:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        return

    banner()
    
    section("🔧 Building AI System — Registering Modules")
    agent = build_system()
    planner = agent._modules['Planner']
    print(f"\n{C.GREEN}✅ System Assembly Complete! All sub-modules active.{C.END}")

    print("\nSelect Mode:")
    print("  [1] Run Standard 5 Test Patients (P001 - P005) & Generate Dashboard")
    print("  [2] Custom Input Mode (Enter Temp, HR, BP, and Symptoms via Terminal)")
    if HAS_STREAMLIT:
        print("  [3] Launch Web Application Interface (Streamlit Browser App)")

    try:
        choice = input("\nEnter choice [1-3] (Default: 1): ").strip()
    except EOFError:
        choice = "1"

    if choice == "2":
        run_interactive_cli(agent, planner)
    elif choice == "3" and HAS_STREAMLIT:
        print(f"\n{C.BLUE}🚀 Launching Web Interface...{C.END}")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        patients = get_sample_patients()
        section("🏥 Running Diagnostics on Sample Patients")
        all_reports = []

        for patient in patients:
            report = agent.run(patient)
            all_reports.append(report)
            display_report(patient, report, planner)

        section("📊 Performance Metrics & System Logs")
        agent.print_log()
        
        perf = agent.get_performance()
        print(f"\n  Total Patients Processed: {perf['total_patients']}")
        print(f"  Diagnoses Completed:     {perf['diagnoses_made']}")
        print(f"  Agent Performance Score: {perf['performance_score']}")

        plot_capstone_dashboard(all_reports)

if __name__ == "__main__":
    main()
