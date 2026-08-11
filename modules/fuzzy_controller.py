# ============================================================
# MODULE 6: Fuzzy Logic — Patient Severity Assessment
# Covers: Week 12 (Fuzzy Logic)
# ============================================================

import numpy as np
from typing import Dict, Any


class FuzzySeverityAssessor:
    """
    Fuzzy logic system for patient severity assessment.
    Inputs:  Temperature, Heart Rate, Symptom Count
    Output:  Severity Score (0-100)
    """

    def _membership_temp(self, temp: float) -> Dict[str, float]:
        """Temperature membership functions (Triangular / Trapezoidal)"""
        return {
            'normal': max(0.0, min(1.0, (37.5 - temp) / 1.0))
                      if temp <= 37.5 else 0.0,
            'mild':   max(0.0, 1.0 - abs(temp - 38.0) / 1.0),
            'high':   max(0.0, 1.0 - abs(temp - 39.0) / 1.0),
            'critical': max(0.0, min(1.0, (temp - 39.0) / 1.5))
                        if temp >= 39.0 else 0.0
        }

    def _membership_hr(self, hr: int) -> Dict[str, float]:
        """Heart rate membership functions"""
        return {
            'low':      max(0.0, min(1.0, (70 - hr) / 10.0))
                        if hr <= 70 else 0.0,
            'normal':   max(0.0, 1.0 - abs(hr - 80) / 20.0),
            'elevated': max(0.0, 1.0 - abs(hr - 100) / 15.0),
            'high':     max(0.0, min(1.0, (hr - 100) / 20.0))
                        if hr >= 100 else 0.0
        }

    def _membership_symptoms(self, count: int) -> Dict[str, float]:
        """Symptom count membership functions"""
        return {
            'few':      max(0.0, min(1.0, (3 - count) / 2.0)),
            'moderate': max(0.0, 1.0 - abs(count - 4) / 2.0),
            'many':     max(0.0, min(1.0, (count - 5) / 3.0))
        }

    def _defuzzify(self, severity_rules: Dict[str, float]) -> float:
        """Centroid defuzzification method"""
        centers = {'low': 15, 'mild': 35, 'moderate': 55,
                   'high': 75, 'critical': 92}
        numerator   = sum(centers[k] * v
                          for k, v in severity_rules.items()
                          if k in centers)
        denominator = sum(severity_rules.values()) + 1e-10
        return numerator / denominator

    def assess(self, temperature: float, heart_rate: int,
               symptom_count: int) -> Dict[str, Any]:
        """Full fuzzy inference pipeline"""
        # Step 1: Fuzzification
        temp_mf    = self._membership_temp(temperature)
        hr_mf      = self._membership_hr(heart_rate)
        symptom_mf = self._membership_symptoms(symptom_count)

        # Step 2: Rule evaluation (min for AND, max for OR)
        rules = {
            'critical': max(
                min(temp_mf['critical'], hr_mf['high']),
                min(temp_mf['critical'], symptom_mf['many'])
            ),
            'high': max(
                min(temp_mf['high'], hr_mf['elevated']),
                min(temp_mf['high'], symptom_mf['many']),
                min(temp_mf['mild'], hr_mf['high'])
            ),
            'moderate': max(
                min(temp_mf['mild'], hr_mf['normal']),
                min(temp_mf['high'], symptom_mf['moderate']),
                min(temp_mf['normal'], symptom_mf['many'])
            ),
            'mild': max(
                min(temp_mf['mild'], symptom_mf['few']),
                min(temp_mf['normal'], symptom_mf['moderate'])
            ),
            'low': min(temp_mf['normal'], hr_mf['normal'],
                       symptom_mf['few'])
        }

        # Step 3: Defuzzification
        severity_score = self._defuzzify(rules)
        severity_label = self._classify(severity_score)

        return {
            'severity_score': round(severity_score, 2),
            'severity_label': severity_label,
            'rule_strengths': {k: round(v, 3) for k, v in rules.items()},
            'memberships': {
                'temperature': temp_mf,
                'heart_rate':  hr_mf,
                'symptoms':    symptom_mf
            }
        }

    def _classify(self, score: float) -> str:
        """Map score ranges to qualitative severity labels"""
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MODERATE"
        elif score >= 20:
            return "MILD"
        return "LOW"

    def analyze(self, percept: Any) -> Dict[str, Any]:
        """Module interface for the agent"""
        result = self.assess(
            percept.temperature,
            percept.heart_rate,
            len(percept.symptoms)
        )
        result['summary']    = (f"Severity: {result['severity_label']} "
                                f"({result['severity_score']:.1f}/100)")
        result['diagnosis']  = result['severity_label']
        result['confidence'] = result['severity_score'] / 100
        return result


# ============================================================
# STANDALONE MODULE VERIFICATION TEST
# ============================================================
if __name__ == "__main__":
    print("\n[OK] Fuzzy severity assessment completed successfully.")
    fa = FuzzySeverityAssessor()

    test_cases = [
        (37.0, 72, 2, "Normal patient"),
        (38.5, 95, 4, "Mild/Moderate illness"),
        (39.2, 115, 6, "Severe case"),
        (40.2, 130, 9, "Critical case"),
    ]

    print(f"{'Description':<22} | {'Temp (°C)':<9} | {'HR':<4} | {'Symp':<4} | {'Score':<6} | {'Label'}")
    print("-" * 70)

    for temp, hr, count, desc in test_cases:
        res = fa.assess(temp, hr, count)
        print(f"{desc:<22} | {temp:<9.1f} | {hr:<4} | {count:<4} | {res['severity_score']:<6.1f} | {res['severity_label']}")

    print("\n[✔] Fuzzy Logic Severity Assessor test passed!")