"""Variable names of the Feynman equations, in the order of experiments/feynman.py."""
NAMES = {
    "I.6.20a": ["θ"], "I.6.20": ["σ", "θ"], "I.6.20b": ["σ", "θ", "θ₁"],
    "I.12.11": ["q", "E_f", "B", "v", "θ"], "I.13.12": ["m₁", "m₂", "r₁", "r₂", "G"],
    "I.29.16": ["x₁", "x₂", "θ₁", "θ₂"], "I.37.4": ["I₁", "I₂", "δ"],
    "I.50.26": ["x₁", "ω", "t", "α"], "II.6.15a": ["ε", "p_d", "r", "x", "y", "z"],
    "I.9.18": ["m₁", "m₂", "x₂", "x₁", "y₂", "y₁", "z₂", "z₁", "G"],
}
LATEX = {"θ": r"\theta", "θ₁": r"\theta_1", "θ₂": r"\theta_2", "σ": r"\sigma", "ω": r"\omega", "α": r"\alpha",
         "δ": r"\delta", "ε": r"\epsilon", "m₁": "m_1", "m₂": "m_2", "r₁": "r_1", "r₂": "r_2", "x₁": "x_1",
         "x₂": "x_2", "y₁": "y_1", "y₂": "y_2", "z₁": "z_1", "z₂": "z_2", "I₁": "I_1", "I₂": "I_2",
         "E_f": "E_f", "p_d": "p_d"}


def names(target):
    if target.startswith("feynman:"):
        return [LATEX.get(v, v) for v in NAMES[target.split(":", 1)[1]]]
    return [f"x_{i + 1}" for i in range(9)]
