from typing import Dict


def get_recommendations(final_phase: str) -> Dict[str, str]:
    recs = {
        "Menstrual": {
            "workout": "Light movement, stretching, walking, low-intensity exercise.",
            "nutrition": "Iron-rich foods, hydration, warm easy-to-digest meals.",
        },
        "Follicular": {
            "workout": "Moderate to higher-energy workouts, strength training, cardio.",
            "nutrition": "Balanced meals with protein, vegetables, and fiber.",
        },
        "Fertility": {
            "workout": "Higher-energy sessions if tolerated, mixed cardio and strength.",
            "nutrition": "Balanced anti-inflammatory meals, hydration, steady energy intake.",
        },
        "Luteal": {
            "workout": "Moderate intensity, Pilates, strength with extra recovery.",
            "nutrition": "Magnesium-rich foods, balanced meals, hydration, blood sugar support.",
        },
    }
    return recs.get(final_phase, {"workout": "", "nutrition": ""})
