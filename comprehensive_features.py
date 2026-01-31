"""
Comprehensive Feature Extraction from QASM Files

This module shows ALL possible features you can extract, from simple to advanced.
Features are organized by difficulty and expected impact.
"""

import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np


class QASMFeatureExtractor:
    """Extract features from OpenQASM 2.0 files."""

    def __init__(self, qasm_path):
        self.path = Path(qasm_path)
        self.text = self.path.read_text(encoding='utf-8')
        self.lines = [ln.strip() for ln in self.text.splitlines()
                      if ln.strip() and not ln.strip().startswith('//')]

    def extract_all(self):
        """Extract all features at once."""
        features = {}
        features.update(self.basic_counts())
        features.update(self.gate_type_features())
        features.update(self.qubit_interaction_features())
        features.update(self.depth_features())
        features.update(self.entanglement_proxy_features())
        features.update(self.structural_patterns())
        features.update(self.derived_ratios())
        return features

    # ==================================================================
    # LEVEL 1: BASIC COUNTS (Easy, surprisingly effective!)
    # ==================================================================

    def basic_counts(self):
        """Most basic features - just count things."""
        features = {}

        # Line counts
        features['n_lines'] = len(self.lines)
        features['n_nonempty_lines'] = sum(1 for ln in self.lines if ln)

        # Qubit counts
        qreg_matches = re.findall(r'qreg\s+\w+\[(\d+)\]', self.text)
        features['n_qubits'] = sum(int(x) for x in qreg_matches)
        features['n_qreg_declarations'] = len(qreg_matches)

        # Classical register counts
        creg_matches = re.findall(r'creg\s+\w+\[(\d+)\]', self.text)
        features['n_classical_bits'] = sum(int(x) for x in creg_matches)

        # Measurement count
        features['n_measure'] = len(re.findall(r'\bmeasure\b', self.text))

        # Barrier count (prevents optimization, might indicate structure)
        features['n_barrier'] = len(re.findall(r'\bbarrier\b', self.text))

        return features

    # ==================================================================
    # LEVEL 2: GATE TYPE FEATURES (Easy, very important!)
    # ==================================================================

    def gate_type_features(self):
        """Count different gate types - KEY for predicting complexity."""
        features = {}

        # 2-QUBIT GATES (create entanglement - most important!)
        features['n_cx'] = len(re.findall(r'\bcx\b', self.text))
        features['n_cz'] = len(re.findall(r'\bcz\b', self.text))
        features['n_cp'] = len(re.findall(r'\bcp\b', self.text))  # controlled phase
        features['n_cy'] = len(re.findall(r'\bcy\b', self.text))  # controlled Y
        features['n_ch'] = len(re.findall(r'\bch\b', self.text))  # controlled H
        features['n_swap'] = len(re.findall(r'\bswap\b', self.text))
        features['n_ccx'] = len(re.findall(r'\bccx\b', self.text))  # Toffoli (3-qubit!)
        features['n_cswap'] = len(re.findall(r'\bcswap\b', self.text))  # Fredkin

        # Total 2-qubit gates
        features['n_2q_gates'] = (features['n_cx'] + features['n_cz'] +
                                  features['n_cp'] + features['n_cy'] +
                                  features['n_ch'] + features['n_swap'])

        # Total 3-qubit gates (very expensive!)
        features['n_3q_gates'] = features['n_ccx'] + features['n_cswap']

        # 1-QUBIT GATES (much cheaper)
        # Pauli gates
        features['n_x'] = len(re.findall(r'\bx\b', self.text))
        features['n_y'] = len(re.findall(r'\by\b', self.text))
        features['n_z'] = len(re.findall(r'\bz\b', self.text))
        features['n_h'] = len(re.findall(r'\bh\b', self.text))  # Hadamard

        # Phase gates
        features['n_s'] = len(re.findall(r'\bs\b', self.text))
        features['n_sdg'] = len(re.findall(r'\bsdg\b', self.text))
        features['n_t'] = len(re.findall(r'\bt\b', self.text))
        features['n_tdg'] = len(re.findall(r'\btdg\b', self.text))

        # Rotation gates (parametric)
        features['n_rx'] = len(re.findall(r'\brx\b', self.text))
        features['n_ry'] = len(re.findall(r'\bry\b', self.text))
        features['n_rz'] = len(re.findall(r'\brz\b', self.text))

        # Universal gates
        features['n_u1'] = len(re.findall(r'\bu1\b', self.text))
        features['n_u2'] = len(re.findall(r'\bu2\b', self.text))
        features['n_u3'] = len(re.findall(r'\bu3\b', self.text))
        features['n_u'] = len(re.findall(r'\bu\(', self.text))  # generic U

        # Total 1-qubit gates
        one_q_gates = [features[k] for k in features if k.startswith('n_')
                      and k not in ['n_cx', 'n_cz', 'n_cp', 'n_cy', 'n_ch',
                                   'n_swap', 'n_ccx', 'n_cswap', 'n_2q_gates', 'n_3q_gates']]
        features['n_1q_gates'] = sum(one_q_gates)

        # Total gates
        features['n_total_gates'] = features['n_1q_gates'] + features['n_2q_gates'] + features['n_3q_gates']

        return features

    # ==================================================================
    # LEVEL 3: QUBIT INTERACTION FEATURES (Medium difficulty, high impact)
    # ==================================================================

    def qubit_interaction_features(self):
        """Build interaction graph from 2-qubit gates."""
        features = {}

        # Extract all 2-qubit gate interactions
        interactions = []

        # Match patterns like: cx q[0],q[1] or cx eval[5],q[0]
        two_qubit_pattern = r'(?:cx|cz|cp|cy|ch|swap)\s+(\w+)\[(\d+)\]\s*,\s*(\w+)\[(\d+)\]'
        matches = re.findall(two_qubit_pattern, self.text)

        # Build edge list
        edges = []
        for reg1, idx1, reg2, idx2 in matches:
            # Create unique qubit identifiers
            q1 = f"{reg1}[{idx1}]"
            q2 = f"{reg2}[{idx2}]"
            edges.append((q1, q2))

        features['n_unique_edges'] = len(set(edges))
        features['n_edge_repetitions'] = len(edges) - features['n_unique_edges']

        if not edges:
            # No interactions - trivial circuit
            features['max_qubit_degree'] = 0
            features['avg_qubit_degree'] = 0.0
            features['qubit_degree_std'] = 0.0
            features['n_connected_components'] = 0
            return features

        # Build adjacency list
        adjacency = defaultdict(set)
        for q1, q2 in edges:
            adjacency[q1].add(q2)
            adjacency[q2].add(q1)

        # Degree statistics
        degrees = [len(neighbors) for neighbors in adjacency.values()]
        features['max_qubit_degree'] = max(degrees) if degrees else 0
        features['avg_qubit_degree'] = np.mean(degrees) if degrees else 0.0
        features['qubit_degree_std'] = np.std(degrees) if degrees else 0.0

        # Count connected components (simplified BFS)
        visited = set()
        n_components = 0
        for node in adjacency:
            if node not in visited:
                # BFS to find component
                queue = [node]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    queue.extend(adjacency[current] - visited)
                n_components += 1

        features['n_connected_components'] = n_components

        return features

    # ==================================================================
    # LEVEL 4: DEPTH FEATURES (Medium difficulty)
    # ==================================================================

    def depth_features(self):
        """Estimate circuit depth (crude approximation)."""
        features = {}

        n_qubits = sum(int(x) for x in re.findall(r'qreg\s+\w+\[(\d+)\]', self.text))

        if n_qubits == 0:
            features['crude_depth'] = 0.0
            features['gates_per_layer_estimate'] = 0.0
            return features

        # Very crude depth estimate: lines / qubits
        # (Assumes roughly uniform distribution of gates)
        features['crude_depth'] = len(self.lines) / n_qubits

        # Estimate gates per layer
        n_total_gates = len(re.findall(r'\b(?:cx|cz|h|x|y|z|s|t|rx|ry|rz|u1|u2|u3)\b', self.text))
        features['gates_per_layer_estimate'] = n_total_gates / features['crude_depth'] if features['crude_depth'] > 0 else 0.0

        return features

    # ==================================================================
    # LEVEL 5: ENTANGLEMENT PROXIES (Medium-hard, physics-inspired)
    # ==================================================================

    def entanglement_proxy_features(self):
        """Features that proxy for entanglement without simulation."""
        features = {}

        n_qubits = sum(int(x) for x in re.findall(r'qreg\s+\w+\[(\d+)\]', self.text))
        n_2q_gates = len(re.findall(r'\b(?:cx|cz|cp|cy|ch|swap)\b', self.text))

        # Average "span" of 2-qubit gates
        # (How far apart are the qubits being entangled?)
        two_qubit_pattern = r'(?:cx|cz|cp|cy|ch|swap)\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]'
        matches = re.findall(two_qubit_pattern, self.text)

        if matches:
            spans = [abs(int(idx1) - int(idx2)) for idx1, idx2 in matches]
            features['avg_gate_span'] = np.mean(spans)
            features['max_gate_span'] = max(spans)
            features['std_gate_span'] = np.std(spans)
        else:
            features['avg_gate_span'] = 0.0
            features['max_gate_span'] = 0
            features['std_gate_span'] = 0.0

        # Entanglement "pressure" - how many gates per qubit
        if n_qubits > 0:
            features['entanglement_pressure'] = n_2q_gates / n_qubits
        else:
            features['entanglement_pressure'] = 0.0

        # Cut-based proxy: if we cut the circuit in half, how many gates cross?
        if n_qubits > 1:
            mid = n_qubits // 2
            crossing_gates = sum(1 for idx1, idx2 in matches
                               if (int(idx1) < mid and int(idx2) >= mid) or
                                  (int(idx1) >= mid and int(idx2) < mid))
            features['midpoint_cut_crossings'] = crossing_gates
        else:
            features['midpoint_cut_crossings'] = 0

        return features

    # ==================================================================
    # LEVEL 6: STRUCTURAL PATTERNS (Pattern recognition)
    # ==================================================================

    def structural_patterns(self):
        """Detect common circuit patterns."""
        features = {}

        # QFT pattern: lots of cp gates with fractional angles
        cp_with_fractions = len(re.findall(r'\bcp\s*\(\s*-?\s*pi\s*/\s*\d+', self.text))
        features['has_qft_pattern'] = 1 if cp_with_fractions > 10 else 0
        features['n_qft_like_gates'] = cp_with_fractions

        # Inverse QFT (common in algorithms)
        features['has_iqft_pattern'] = 1 if 'iqft' in self.text.lower() else 0

        # Grover pattern: lots of x gates and ccx
        n_x = len(re.findall(r'\bx\b', self.text))
        n_ccx = len(re.findall(r'\bccx\b', self.text))
        features['has_grover_pattern'] = 1 if (n_ccx > 5 and n_x > 10) else 0

        # Variational pattern: rx, ry, rz gates (VQE, QAOA)
        n_rotations = len(re.findall(r'\b(?:rx|ry|rz)\b', self.text))
        features['has_variational_pattern'] = 1 if n_rotations > 5 else 0
        features['n_rotation_gates'] = n_rotations

        # GHZ/W-state pattern: regular structure with h and cx
        n_h = len(re.findall(r'\bh\b', self.text))
        n_cx = len(re.findall(r'\bcx\b', self.text))
        features['has_ghz_pattern'] = 1 if (n_h > 0 and n_cx > 3 and len(self.lines) < 50) else 0

        # Custom gate definitions (might be complex)
        features['n_custom_gates'] = len(re.findall(r'\bgate\s+\w+', self.text))

        # Opaque gates (black boxes - unpredictable)
        features['n_opaque_gates'] = len(re.findall(r'\bopaque\s+\w+', self.text))

        return features

    # ==================================================================
    # LEVEL 7: DERIVED RATIOS (Normalize for circuit size)
    # ==================================================================

    def derived_ratios(self):
        """Compute ratios and normalized features."""
        features = {}

        n_qubits = sum(int(x) for x in re.findall(r'qreg\s+\w+\[(\d+)\]', self.text))
        n_2q_gates = len(re.findall(r'\b(?:cx|cz|cp|cy|ch|swap)\b', self.text))
        n_1q_gates = len(re.findall(r'\b(?:h|x|y|z|s|sdg|t|tdg|rx|ry|rz|u1|u2|u3)\b', self.text))
        n_total_gates = n_1q_gates + n_2q_gates

        # Ratio of 2Q to total gates (key metric!)
        if n_total_gates > 0:
            features['ratio_2q_gates'] = n_2q_gates / n_total_gates
            features['ratio_1q_gates'] = n_1q_gates / n_total_gates
        else:
            features['ratio_2q_gates'] = 0.0
            features['ratio_1q_gates'] = 0.0

        # Gates per qubit
        if n_qubits > 0:
            features['gates_per_qubit'] = n_total_gates / n_qubits
            features['2q_gates_per_qubit'] = n_2q_gates / n_qubits
            features['1q_gates_per_qubit'] = n_1q_gates / n_qubits
        else:
            features['gates_per_qubit'] = 0.0
            features['2q_gates_per_qubit'] = 0.0
            features['1q_gates_per_qubit'] = 0.0

        # Circuit density
        if n_qubits > 0 and len(self.lines) > 0:
            features['circuit_density'] = n_total_gates / (n_qubits * len(self.lines))
        else:
            features['circuit_density'] = 0.0

        return features


# ==================================================================
# DEMONSTRATION
# ==================================================================

if __name__ == "__main__":
    import pandas as pd

    print("="*80)
    print("COMPREHENSIVE FEATURE EXTRACTION DEMONSTRATION")
    print("="*80)
    print()

    # Extract from the example circuit
    extractor = QASMFeatureExtractor("circuits/ae_indep_qiskit_20.qasm")
    features = extractor.extract_all()

    print(f"Circuit: ae_indep_qiskit_20.qasm")
    print(f"Total features extracted: {len(features)}")
    print()

    # Organize by category
    categories = {
        'Basic Counts': ['n_lines', 'n_qubits', 'n_measure', 'n_barrier'],
        'Gate Counts': ['n_cx', 'n_cz', 'n_cp', 'n_h', 'n_x', 'n_1q_gates', 'n_2q_gates', 'n_total_gates'],
        'Interaction Graph': ['n_unique_edges', 'max_qubit_degree', 'avg_qubit_degree', 'n_connected_components'],
        'Depth': ['crude_depth', 'gates_per_layer_estimate'],
        'Entanglement Proxies': ['avg_gate_span', 'max_gate_span', 'entanglement_pressure', 'midpoint_cut_crossings'],
        'Patterns': ['has_qft_pattern', 'has_variational_pattern', 'has_grover_pattern', 'n_custom_gates'],
        'Ratios': ['ratio_2q_gates', 'gates_per_qubit', '2q_gates_per_qubit', 'circuit_density'],
    }

    for category, feature_names in categories.items():
        print(f"--- {category} ---")
        for fname in feature_names:
            if fname in features:
                value = features[fname]
                if isinstance(value, float):
                    print(f"  {fname:30s}: {value:>10.3f}")
                else:
                    print(f"  {fname:30s}: {value:>10}")
        print()

    print("="*80)
    print("FEATURE IMPORTANCE GUIDE")
    print("="*80)
    print()
    print("🔥 HIGH IMPACT (use these first!):")
    print("  • n_qubits - circuit size is critical")
    print("  • n_2q_gates - entangling gates drive complexity")
    print("  • ratio_2q_gates - high ratio = high entanglement")
    print("  • gates_per_qubit - circuit density")
    print("  • max_qubit_degree - connectivity matters")
    print()
    print("⭐ MEDIUM IMPACT (add these next):")
    print("  • avg_gate_span - long-range gates are harder")
    print("  • n_connected_components - fragmented circuits")
    print("  • crude_depth - longer circuits need more resources")
    print("  • entanglement_pressure - gates per qubit ratio")
    print()
    print("💡 NICE TO HAVE (diminishing returns):")
    print("  • Pattern features (QFT, Grover, etc.)")
    print("  • Specific gate counts (n_rx, n_ry, etc.)")
    print("  • Edge statistics")
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Run this on ALL circuits to build feature matrix")
    print("2. Combine with backend/precision config features")
    print("3. Train models using top features")
    print("4. Check feature importance from trained models")
    print("5. Iterate - add/remove features based on what works")
    print()

    # Show how to extract for all circuits
    print("="*80)
    print("EXTRACTING FOR ALL CIRCUITS")
    print("="*80)
    print()

    circuit_dir = Path("circuits")
    all_data = []

    for qasm_file in sorted(circuit_dir.glob("*.qasm"))[:5]:  # First 5 for demo
        ext = QASMFeatureExtractor(qasm_file)
        feats = ext.extract_all()
        feats['filename'] = qasm_file.name
        all_data.append(feats)

    df = pd.DataFrame(all_data)
    print("Feature matrix shape:", df.shape)
    print()
    print("Top features preview:")
    print(df[['filename', 'n_qubits', 'n_2q_gates', 'ratio_2q_gates', 'gates_per_qubit']].to_string())
    print()
    print(f"... extracting from all {len(list(circuit_dir.glob('*.qasm')))} circuits")
    print()
