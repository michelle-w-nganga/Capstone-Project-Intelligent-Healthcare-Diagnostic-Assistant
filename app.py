# ============================================================  
# CAPSTONE MAIN APPLICATION  
# Intelligent Healthcare Diagnostic Assistant  
# Introduction to AI — 13-Week Capstone  
# ============================================================  

import sys  
import json  
import warnings  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
warnings.filterwarnings('ignore')  

# Import all modules  
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
    section("🔧 Building AI System — Registering Modules")  

    agent = HealthcareDiagnosticAgent()  

    print("\n  Initializing modules...")  
    
    # 1. Logic Knowledge Base
    kb = MedicalKnowledgeBase()
    agent.register_module('KnowledgeBase', kb)

    # 2. Bayesian Network
    bn = SimpleBayesianDiagnostics()
    agent.register_module('BayesianNet', bn)

    # 3. ML Classifier (Train before attaching)
    ml = MLDiagnosticClassifier()
    ml.train(verbose=False)
    agent.register_module('MLClassifier', ml)

    # 4. Neural Network (Train before attaching)
    nn = NeuralDiagnosticModel()
    nn.train(epochs=25, verbose=0)
    agent.register_module('NeuralNetwork', nn)

    # 5. Fuzzy Logic Controller
    fuzzy = FuzzySeverityAssessor()
    agent.register_module('FuzzySeverity', fuzzy)

    # 6. AI Treatment Planner
    planner = TreatmentPlanner()
    agent.register_module('Planner', planner)

    print(f"\n{C.GREEN}✅ System Assembly Complete! All sub-modules active.{C.END}")
    return agent

def get_sample_patients():
    """Returns test cases covering various diseases and clinical severities"""
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

    # Generate and display treatment plan
    tx_plan = planner.create_treatment_plan(report['diagnosis'], report['urgency'])
    print(f"\n  {C.BOLD}Generated Treatment Plan ({tx_plan.get('steps', 0)} steps):{C.END}")
    for step in tx_plan.get('plan', []):
        print(f"    Step {step['step']}: {step['action']:<25} [{step['duration']}]")

def plot_capstone_dashboard(all_reports: list):
    """Plots combined evaluation metrics and exports final dashboard image"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # 1. Diagnoses Summary
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

    # 2. Urgency Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    urgencies = [r['urgency'] for r in all_reports]
    u_counts = {u: urgencies.count(u) for u in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']}
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    
    ax2.pie([v for v in u_counts.values() if v > 0], 
            labels=[k for k, v in u_counts.items() if v > 0], 
            autopct='%1.1f%%', colors=[c for k, c in zip(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], colors) if u_counts[k] > 0],
            startangle=140)
    ax2.set_title("Patient Urgency Triage Distribution", fontweight='bold')

    # 3. Sub-Module Agreement Map
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

def main():
    banner()
    
    # Build System
    agent = build_system()
    planner = agent._modules['Planner']
    
    # Load Patients
    patients = get_sample_patients()
    
    section("🏥 Running Live Diagnostics on Test Patients")
    all_reports = []

    # Run Loop
    for patient in patients:
        report = agent.run(patient)
        all_reports.append(report)
        display_report(patient, report, planner)

    # System Logs & Performance
    section("📊 Performance Metrics & System Logs")
    agent.print_log()
    
    perf = agent.get_performance()
    print(f"\n  Total Patients Processed: {perf['total_patients']}")
    print(f"  Diagnoses Completed:     {perf['diagnoses_made']}")
    print(f"  Agent Performance Score: {perf['performance_score']}")

    # Dashboard Generation
    plot_capstone_dashboard(all_reports)

if __name__ == "__main__":
    main()