# ============================================================
# MODULE 1: Intelligent Agent — Healthcare Diagnostic Agent
# Covers: Week 2 (Intelligent Agents) + PEAS Framework
# ============================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime
from collections import Counter


class AgentState(Enum):
    IDLE         = "idle"
    COLLECTING   = "collecting_symptoms"
    DIAGNOSING   = "diagnosing"
    RECOMMENDING = "recommending"
    PLANNING     = "planning_treatment"
    DONE         = "done"


@dataclass
class PatientPercept:
    """What the agent perceives from the environment"""
    patient_id:     str
    symptoms:       List[str]
    age:            int
    temperature:    float
    heart_rate:     int
    blood_pressure: str
    timestamp:      str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )


@dataclass
class AgentMemory:
    """Internal model — makes this a model-based agent"""
    patient_history:   List[Dict]           = field(default_factory=list)
    current_patient:   Optional[PatientPercept] = None
    diagnosis_history: List[Dict]           = field(default_factory=list)
    action_log:        List[str]            = field(default_factory=list)


class HealthcareDiagnosticAgent:
    """
    PEAS Definition:
    ─────────────────────────────────────────────────
    Performance : Diagnostic accuracy, patient safety,
                  recommendation quality, response time
    Environment : Hospital/clinic, patient data, EMR
    Actuators   : Diagnosis report, treatment plan,
                  referral recommendation, alerts
    Sensors     : Symptom input, vitals, lab results,
                  patient history
    ─────────────────────────────────────────────────
    Agent Type  : Model-Based + Goal-Based + Learning
    """

    def __init__(self):
        self.state   = AgentState.IDLE
        self.memory  = AgentMemory()
        self.performance_score = 0
        self._modules = {}  # Holds registered sub-modules

    def register_module(self, name: str, module):
        """Plug in AI sub-modules (KB, Bayes, ML, Fuzzy, Planner, etc.)"""
        self._modules[name] = module
        print(f"  🔌 Module registered: [{name}]")

    def perceive(self, percept: PatientPercept):
        """Step 1: Perceive the environment"""
        self.memory.current_patient = percept
        self.memory.patient_history.append({
            'id': percept.patient_id,
            'symptoms': percept.symptoms,
            'time': percept.timestamp
        })
        self.state = AgentState.COLLECTING
        self._log(f"Perceived patient {percept.patient_id} "
                  f"with {len(percept.symptoms)} symptoms")
        return self

    def think(self) -> Dict:
        """Step 2: Process and reason across all registered sub-modules"""
        if not self.memory.current_patient:
            raise ValueError("No patient percept stored. Call perceive() first.")

        self.state = AgentState.DIAGNOSING
        self._log("Agent thinking: running diagnostic modules...")

        results = {}

        # Run each registered module
        for module_name, module in self._modules.items():
            if hasattr(module, 'analyze'):
                try:
                    result = module.analyze(self.memory.current_patient)
                    results[module_name] = result
                    summary = result.get('summary', 'done') if isinstance(result, dict) else 'done'
                    self._log(f"  [{module_name}] → {summary}")
                except Exception as e:
                    results[module_name] = {"error": str(e)}
                    self._log(f"  [{module_name}] → Error: {e}")

        self.memory.diagnosis_history.append(results)
        self.state = AgentState.RECOMMENDING
        return results

    def act(self, diagnosis_results: Dict) -> Dict:
        """Step 3: Generate action/recommendation and invoke planner if available"""
        if not self.memory.current_patient:
            raise ValueError("No patient percept available to act upon.")

        self.state = AgentState.PLANNING
        patient = self.memory.current_patient

        # Aggregate confidence scores from multiple modules
        confidences = [
            v.get('confidence', 0)
            for v in diagnosis_results.values()
            if isinstance(v, dict) and 'confidence' in v
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Aggregate primary diagnosis
        primary_diagnosis = self._aggregate_diagnosis(diagnosis_results)

        # Determine urgency level based on vitals and overall confidence
        urgency = self._assess_urgency(patient, avg_confidence)

        # Generate action report
        action_report = {
            'patient_id':      patient.patient_id,
            'timestamp':       patient.timestamp,
            'symptoms':        patient.symptoms,
            'diagnosis':       primary_diagnosis,
            'confidence':      round(avg_confidence, 3),
            'urgency':          urgency,
            'recommendations': self._generate_recommendations(urgency, diagnosis_results),
            'next_action':     self._decide_next_action(urgency),
            'module_results':  diagnosis_results
        }

        # Update learning performance metric
        self.performance_score += (10 if avg_confidence > 0.7 else 5)
        self.state = AgentState.DONE
        self._log(f"Action generated: {urgency} urgency for diagnosis '{primary_diagnosis}'")
        return action_report

    def run(self, percept: PatientPercept) -> Dict:
        """Full agent cycle execution: Perceive → Think → Act"""
        self.perceive(percept)
        results = self.think()
        return self.act(results)

    def _assess_urgency(self, patient: PatientPercept, confidence: float) -> str:
        """Rule-based clinical triage using vitals and confidence metrics"""
        if patient.temperature >= 39.5 or patient.heart_rate >= 120:
            return "CRITICAL"
        elif patient.temperature >= 38.5 or confidence > 0.8:
            return "HIGH"
        elif patient.temperature >= 37.5:
            return "MEDIUM"
        return "LOW"

    def _aggregate_diagnosis(self, results: Dict) -> str:
        """Selects top diagnosis using majority voting across sub-modules"""
        diagnoses = [
            v.get('diagnosis', 'Unknown')
            for v in results.values()
            if isinstance(v, dict) and 'diagnosis' in v
        ]
        if not diagnoses:
            return "Insufficient data"
        return Counter(diagnoses).most_common(1)[0][0]

    def _generate_recommendations(self, urgency: str, results: Dict) -> List[str]:
        """Maps urgency tiers to standard clinical action protocols"""
        base = {
            "CRITICAL": [
                "🚨 Immediate emergency consultation required",
                "📞 Alert attending physician now",
                "🏥 Transfer to emergency ward",
                "💊 Administer first-line stabilization protocol"
            ],
            "HIGH": [
                "⚠️ Schedule urgent appointment within 24 hours",
                "🧪 Order blood panel and laboratory diagnostic cultures",
                "💊 Prescribe targeted symptomatic medication",
                "📋 Monitor vitals every 2 hours"
            ],
            "MEDIUM": [
                "📅 Schedule general clinical appointment within 3 days",
                "💊 Over-the-counter therapeutic treatment advised",
                "🌡️ Monitor core temperature twice daily",
                "💧 Maintain proper fluid hydration"
            ],
            "LOW": [
                "🏠 Home rest recommended",
                "💧 Stay well hydrated",
                "📱 Follow up if secondary symptoms manifest",
                "📋 General wellness observation"
            ]
        }
        return base.get(urgency, base["LOW"])

    def _decide_next_action(self, urgency: str) -> str:
        """Decides final system routing based on triage outcome"""
        actions = {
            "CRITICAL": "EMERGENCY_REFERRAL",
            "HIGH":     "URGENT_APPOINTMENT",
            "MEDIUM":   "SCHEDULE_FOLLOWUP",
            "LOW":      "MONITOR_AT_HOME"
        }
        return actions.get(urgency, "MONITOR_AT_HOME")

    def _log(self, message: str) -> None:
        """Internal logger tracking state transitions and module actions"""
        entry = f"[{self.state.value}] {message}"
        self.memory.action_log.append(entry)

    def print_log(self) -> None:
        """Outputs current session interaction logs"""
        print("\n📋 Agent Action Log:")
        print("─" * 50)
        for entry in self.memory.action_log:
            print(f"  {entry}")

    def get_performance(self) -> Dict:
        """Returns agent operational statistics"""
        return {
            'total_patients':    len(self.memory.patient_history),
            'performance_score': self.performance_score,
            'diagnoses_made':    len(self.memory.diagnosis_history)
        }


# ============================================================
# STANDALONE MODULE VERIFICATION TEST
# ============================================================
if __name__ == "__main__":
    print("--- Running Module 1 Test ---")

    # Mock Diagnostic Sub-module for testing pipeline
    class DummySubModule:
        def analyze(self, percept: PatientPercept):
            return {
                'diagnosis': 'covid19',
                'confidence': 0.88,
                'summary': 'High likelihood of respiratory viral infection'
            }

    # 1. Instantiate Agent
    agent = HealthcareDiagnosticAgent()

    # 2. Register Sub-modules
    agent.register_module('KnowledgeBase', DummySubModule())

    # 3. Create Sample Patient Data
    patient_p001 = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough", "loss_of_smell", "fatigue"],
        age=34,
        temperature=38.9,
        heart_rate=98,
        blood_pressure="120/80"
    )

    # 4. Execute Full Agent Execution Cycle (Perceive -> Think -> Act)
    final_output = agent.run(patient_p001)

    # 5. Display System Log and Results
    agent.print_log()

    print("\n--- Output Triage Report ---")
    print(f"Patient ID:        {final_output['patient_id']}")
    print(f"Aggregated Diagnosis: {final_output['diagnosis']}")
    print(f"Mean Confidence:      {final_output['confidence']:.2%}")
    print(f"Urgency Level:        {final_output['urgency']}")
    print(f"Next System Action:   {final_output['next_action']}")
    print("\n[✔] Yay it worked, let's party!!!!!!!!!")