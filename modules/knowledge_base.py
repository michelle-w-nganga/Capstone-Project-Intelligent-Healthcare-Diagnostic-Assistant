# ============================================================
# MODULE 2: FOL Knowledge Base + Inference Engine
# Covers: Week 5 (First-Order Logic & Inference)
# ============================================================

from typing import Set, List, Dict, Tuple, Optional


class MedicalKnowledgeBase:
    """
    First-Order Logic based medical knowledge base.
    Supports forward chaining, backward chaining,
    and confidence-weighted inference.
    """

    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Tuple[List[str], str, float]] = []
        self.certainty_factors: Dict[str, float] = {}
        self._load_medical_knowledge()

    def _load_medical_knowledge(self):
        """Load domain medical knowledge rules"""
        disease_rules = [
            # (conditions, conclusion, certainty)
            (["fever", "cough", "fatigue"],
             "flu_suspected", 0.75),
            (["fever", "cough", "loss_of_smell", "fatigue"],
             "covid19_suspected", 0.85),
            (["fever", "rash", "joint_pain"],
             "dengue_suspected", 0.80),
            (["chest_pain", "shortness_of_breath", "sweating"],
             "cardiac_event_suspected", 0.90),
            (["headache", "stiff_neck", "high_fever", "light_sensitivity"],
             "meningitis_suspected", 0.88),
            (["cough", "weight_loss", "night_sweats", "fatigue"],
             "tuberculosis_suspected", 0.82),
            (["frequent_urination", "excessive_thirst", "blurred_vision"],
             "diabetes_suspected", 0.78),
            (["flu_suspected", "high_fever"],
             "flu_confirmed", 0.85),
            (["covid19_suspected", "positive_pcr"],
             "covid19_confirmed", 0.99),
            (["cardiac_event_suspected", "elevated_troponin"],
             "myocardial_infarction", 0.95),
            # Urgency rules
            (["myocardial_infarction"],
             "EMERGENCY", 1.00),
            (["meningitis_suspected"],
             "EMERGENCY", 0.95),
            (["covid19_confirmed"],
             "ISOLATE_AND_TREAT", 0.99),
            (["flu_confirmed"],
             "REST_AND_MEDICATE", 0.90),
        ]
        for conditions, conclusion, cf in disease_rules:
            self.add_rule(conditions, conclusion, cf)

    def add_fact(self, fact: str, certainty: float = 1.0):
        """Adds a clean fact to the knowledge base with its certainty factor"""
        clean_fact = fact.lower().strip().replace(' ', '_')
        self.facts.add(clean_fact)
        self.certainty_factors[clean_fact] = certainty

    def add_rule(self, conditions: List[str], conclusion: str, certainty: float = 1.0):
        """Adds an IF-THEN rule to the rule base"""
        clean_conditions = [c.lower().strip().replace(' ', '_') for c in conditions]
        clean_conclusion = conclusion.lower().strip().replace(' ', '_')
        self.rules.append((clean_conditions, clean_conclusion, certainty))

    def load_patient_symptoms(self, symptoms: List[str]):
        """Load patient symptoms as facts into memory"""
        for symptom in symptoms:
            self.add_fact(symptom)

    def forward_chain(self, verbose: bool = False) -> Dict[str, float]:
        """
        Forward Chaining with Certainty Factors:
        Iteratively evaluates rules and draws new inferences until no new conclusions can be reached.
        """
        inferred = {}
        changed = True
        iteration = 0

        while changed:
            changed = False
            iteration += 1
            for conditions, conclusion, rule_cf in self.rules:
                all_known = all(
                    c in self.facts or c in inferred for c in conditions
                )
                if all_known and conclusion not in inferred:
                    # Combine certainty factors: rule_cf * min(condition certainty factors)
                    cond_cfs = [
                        self.certainty_factors.get(c, inferred.get(c, 1.0))
                        for c in conditions
                    ]
                    combined_cf = rule_cf * min(cond_cfs)
                    inferred[conclusion] = round(combined_cf, 4)

                    if verbose:
                        cond_str = " ∧ ".join(conditions)
                        print(f"  Iter {iteration}: "
                              f"{cond_str} → {conclusion} "
                              f"(CF={combined_cf:.3f})")
                    changed = True
        return inferred

    def backward_chain(self, goal: str,
                       visited: Optional[Set[str]] = None,
                       depth: int = 0) -> Tuple[bool, float]:
        """
        Backward Chaining:
        Goal-driven approach to check if a target hypothesis/diagnosis can be proved from current facts.
        """
        goal_clean = goal.lower().strip().replace(' ', '_')
        visited = visited or set()

        if goal_clean in self.facts:
            return True, self.certainty_factors.get(goal_clean, 1.0)
        if goal_clean in visited:
            return False, 0.0
        visited.add(goal_clean)

        for conditions, conclusion, rule_cf in self.rules:
            if conclusion == goal_clean:
                results = [
                    self.backward_chain(c, visited.copy(), depth + 1)
                    for c in conditions
                ]
                if all(proved for proved, _ in results):
                    cf = rule_cf * min(cf for _, cf in results)
                    return True, round(cf, 4)
        return False, 0.0

    def analyze(self, percept) -> Dict:
        """Standard interface method invoked by Module 1 Agent"""
        self.facts = set()
        self.certainty_factors = {}
        self.load_patient_symptoms(percept.symptoms)

        # Map clinical vitals to logic facts
        if percept.temperature > 38.0:
            self.add_fact("fever", min(1.0, (percept.temperature - 37.0) / 3.0))
        if percept.temperature > 39.5:
            self.add_fact("high_fever", 1.0)
        if percept.heart_rate > 100:
            self.add_fact("tachycardia", 1.0)

        inferred = self.forward_chain()
        diseases = {k: v for k, v in inferred.items()
                    if 'suspected' in k or 'confirmed' in k}

        top = max(diseases, key=diseases.get) if diseases else "Unknown"
        return {
            'summary':    f"Inferred {len(inferred)} conclusions",
            'diagnosis':  top.replace('_suspected', '').replace('_confirmed', ''),
            'confidence': diseases.get(top, 0.5),
            'all_inferred': inferred
        }

    def get_explanation(self, diagnosis: str) -> str:
        """Explains how a specific diagnosis was inferred"""
        diag_clean = diagnosis.lower().strip().replace(' ', '_')
        for conditions, conclusion, cf in self.rules:
            if conclusion == diag_clean:
                return (f"'{diagnosis}' derived from: "
                        f"{' + '.join(conditions)} (CF={cf})")
        return f"'{diagnosis}' is a base fact"


# ============================================================
# STANDALONE MODULE VERIFICATION TEST
# ============================================================
if __name__ == "__main__":
    print("--- Running Module 2 Test ---")
    kb = MedicalKnowledgeBase()

    # Load symptoms manually
    kb.add_fact("fever", 0.9)
    kb.add_fact("cough", 1.0)
    kb.add_fact("loss_of_smell", 1.0)
    kb.add_fact("fatigue", 1.0)

    # Test Forward Chaining
    print("\n[Forward Chaining Execution]:")
    results = kb.forward_chain(verbose=True)
    print("Inferred conclusions:", results)

    # Test Backward Chaining
    print("\n[Backward Chaining Execution]:")
    proved, cf = kb.backward_chain("covid19_suspected")
    print(f"Goal 'covid19_suspected' Proved: {proved}, Certainty Factor: {cf}")

    print("\n[✔] Knowledge Base test passed!")