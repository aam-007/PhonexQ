"""
PhonexQ - High-Performance Quantum Circuit Simulator
Python Frontend with C Backend Accelerator
"""

import ctypes
import math
import random
import os
import sys
from typing import List, Tuple, Optional

# ==============================================================================
# CONFIGURATION & LIBRARY LOADING
# ==============================================================================

def _load_phonex_library():
    """
    Attempts to load the compiled C library with platform-specific naming.
    Searches common paths and extensions (.so for Linux/Unix, .dll for Windows, .dylib for macOS)
    """
    system = sys.platform
    if system == "win32":
        lib_names = ["phonexq_core.dll", "libphonexq_core.dll"]
    elif system == "darwin":
        lib_names = ["phonexq_core.dylib", "libphonexq_core.dylib", "phonexq_core.so"]
    else:  # Linux and others
        lib_names = ["phonexq_core.so", "libphonexq_core.so"]
    
    # Search paths: current directory, ../lib, system PATH
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        "."
    ]
    
    for path in search_paths:
        for name in lib_names:
            full_path = os.path.join(path, name)
            if os.path.exists(full_path):
                try:
                    return ctypes.CDLL(full_path)
                except OSError as e:
                    print(f"Warning: Found {full_path} but failed to load: {e}")
                    continue
    
    raise RuntimeError(
        f"Could not find PhonexQ core library. "
        f"Searched for: {lib_names} in {search_paths}. "
        f"Please compile the C library first."
    )

# Load the shared library
try:
    C_LIB = _load_phonex_library()
except RuntimeError as e:
    print(f"Error: {e}")
    sys.exit(1)

# ==============================================================================
# C-TYPES DEFINITIONS
# ==============================================================================

class Complex(ctypes.Structure):
    """Matches C struct for complex numbers (double real, double imag)"""
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    @classmethod
    def from_complex(cls, c: complex) -> 'Complex':
        """Convert Python complex to C Complex"""
        return cls(c.real, c.imag)
    
    def to_complex(self) -> complex:
        """Convert C Complex to Python complex"""
        return complex(self.real, self.imag)


class QuantumSystem(ctypes.Structure):
    """Opaque handle to quantum state in C memory"""
    _fields_ = [
        ("num_qubits", ctypes.c_uint),
        ("state_size", ctypes.c_size_t),
        ("state_vector", ctypes.POINTER(Complex))
    ]


# ==============================================================================
# FUNCTION SIGNATURES (CRITICAL FOR TYPE SAFETY)
# ==============================================================================

# System Lifecycle
C_LIB.init_system.argtypes = [ctypes.c_uint]
C_LIB.init_system.restype = ctypes.POINTER(QuantumSystem)

C_LIB.free_system.argtypes = [ctypes.POINTER(QuantumSystem)]
C_LIB.free_system.restype = None

# Gate Operations
C_LIB.apply_gate.argtypes = [
    ctypes.POINTER(QuantumSystem), 
    ctypes.c_uint,  # target
    Complex,        # u00
    Complex,        # u01
    Complex,        # u10
    Complex         # u11
]
C_LIB.apply_gate.restype = None

C_LIB.apply_controlled_gate.argtypes = [
    ctypes.POINTER(QuantumSystem),
    ctypes.c_uint,  # control
    ctypes.c_uint,  # target
    Complex,        # u00
    Complex,        # u01
    Complex,        # u10
    Complex         # u11
]
C_LIB.apply_controlled_gate.restype = None

# Measurement and Probabilities
C_LIB.get_probability.argtypes = [ctypes.POINTER(QuantumSystem), ctypes.c_size_t]
C_LIB.get_probability.restype = ctypes.c_double

C_LIB.measure_qubit.argtypes = [
    ctypes.POINTER(QuantumSystem), 
    ctypes.c_uint,      # target qubit
    ctypes.c_double     # random value 0.0-1.0 (for deterministic testing)
]
C_LIB.measure_qubit.restype = ctypes.c_int

# State inspection (if available in C library)
# C_LIB.get_amplitude.argtypes = [ctypes.POINTER(QuantumSystem), ctypes.c_size_t]
# C_LIB.get_amplitude.restype = Complex

# ==============================================================================
# QUANTUM GATES (Unitary Matrices)
# ==============================================================================

INV_SQRT2 = 1.0 / math.sqrt(2)

# Gate matrices stored as tuples (u00, u01, u10, u11)
GATES = {
    'X': (0, 1, 1, 0),                    # Pauli-X (NOT)
    'Y': (0, -1j, 1j, 0),                 # Pauli-Y
    'Z': (1, 0, 0, -1),                   # Pauli-Z (Phase flip)
    'H': (INV_SQRT2, INV_SQRT2,           # Hadamard
          INV_SQRT2, -INV_SQRT2),
    'I': (1, 0, 0, 1),                    # Identity
    'S': (1, 0, 0, 1j),                   # Phase (optional)
    'T': (1, 0, 0, complex(INV_SQRT2, INV_SQRT2))  # T-gate (optional)
}

# ==============================================================================
# MAIN SIMULATOR CLASS
# ==============================================================================

class PhonexQ:
    """
    High-performance quantum circuit simulator using C backend.
    
    Manages quantum state vector and provides gate-level operations.
    Automatically handles memory management via Python's garbage collection.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize quantum system with given number of qubits.
        
        Args:
            num_qubits: Number of qubits in the system (1-20 recommended)
            
        Raises:
            RuntimeError: If C library fails to initialize system
            ValueError: If num_qubits is invalid
        """
        if not (1 <= num_qubits <= 30):
            raise ValueError("num_qubits must be between 1 and 30")
            
        self.num_qubits = num_qubits
        self.sys_ptr = C_LIB.init_system(num_qubits)
        
        if not self.sys_ptr:
            raise RuntimeError("Failed to initialize quantum system in C library")
            
        self.circuit_log: List[str] = []
        self._measurement_history: List[Tuple[int, int]] = []  # (qubit, result)

    def __del__(self):
        """Ensure C memory is freed when Python object is garbage collected"""
        if hasattr(self, 'sys_ptr') and self.sys_ptr:
            try:
                C_LIB.free_system(self.sys_ptr)
                self.sys_ptr = None
            except Exception:
                # Prevent errors during interpreter shutdown
                pass

    def _validate_qubit(self, qubit: int) -> None:
        """Validate qubit index is within range"""
        if not (0 <= qubit < self.num_qubits):
            raise IndexError(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")

    def _apply_kernel(self, target: int, kernel: Tuple[complex, complex, complex, complex]) -> None:
        """Internal: Apply single-qubit unitary kernel"""
        self._validate_qubit(target)
        
        u00 = Complex.from_complex(kernel[0])
        u01 = Complex.from_complex(kernel[1])
        u10 = Complex.from_complex(kernel[2])
        u11 = Complex.from_complex(kernel[3])
        
        C_LIB.apply_gate(self.sys_ptr, target, u00, u01, u10, u11)

    def _apply_controlled_kernel(self, control: int, target: int, 
                                  kernel: Tuple[complex, complex, complex, complex]) -> None:
        """Internal: Apply controlled-unitary operation"""
        self._validate_qubit(control)
        self._validate_qubit(target)
        
        if control == target:
            raise ValueError("Control and target qubits must be different")
            
        u00 = Complex.from_complex(kernel[0])
        u01 = Complex.from_complex(kernel[1])
        u10 = Complex.from_complex(kernel[2])
        u11 = Complex.from_complex(kernel[3])
        
        C_LIB.apply_controlled_gate(self.sys_ptr, control, target, u00, u01, u10, u11)

    # -------------------------------------------------------------------------
    # Single-Qubit Gates
    # -------------------------------------------------------------------------
    
    def x(self, target: int) -> 'PhonexQ':
        """Pauli-X gate (NOT) on target qubit. Returns self for chaining."""
        self._apply_kernel(target, GATES['X'])
        self.circuit_log.append(f"X({target})")
        return self

    def y(self, target: int) -> 'PhonexQ':
        """Pauli-Y gate on target qubit."""
        self._apply_kernel(target, GATES['Y'])
        self.circuit_log.append(f"Y({target})")
        return self

    def z(self, target: int) -> 'PhonexQ':
        """Pauli-Z gate (phase flip) on target qubit."""
        self._apply_kernel(target, GATES['Z'])
        self.circuit_log.append(f"Z({target})")
        return self

    def h(self, target: int) -> 'PhonexQ':
        """Hadamard gate (creates superposition) on target qubit."""
        self._apply_kernel(target, GATES['H'])
        self.circuit_log.append(f"H({target})")
        return self

    def i(self, target: int) -> 'PhonexQ':
        """Identity gate (no-op) on target qubit."""
        self._apply_kernel(target, GATES['I'])
        self.circuit_log.append(f"I({target})")
        return self

    # -------------------------------------------------------------------------
    # Multi-Qubit Gates
    # -------------------------------------------------------------------------
    
    def cnot(self, control: int, target: int) -> 'PhonexQ':
        """Controlled-NOT gate (CX). Flips target if control is |1>."""
        self._apply_controlled_kernel(control, target, GATES['X'])
        self.circuit_log.append(f"CNOT({control},{target})")
        return self

    def cz(self, control: int, target: int) -> 'PhonexQ':
        """Controlled-Z gate (CZ). Applies phase flip to target if control is |1>."""
        # CZ = Controlled-Z
        # Can be implemented as H(target) -> CNOT(control, target) -> H(target)
        self.h(target)
        self.cnot(control, target)
        self.h(target)
        self.circuit_log.append(f"CZ({control},{target})")
        return self

    def swap(self, qubit1: int, qubit2: int) -> 'PhonexQ':
        """SWAP gate. Exchanges states of two qubits using 3 CNOTs."""
        self.cnot(qubit1, qubit2)
        self.cnot(qubit2, qubit1)
        self.cnot(qubit1, qubit2)
        self.circuit_log.append(f"SWAP({qubit1},{qubit2})")
        return self

    # -------------------------------------------------------------------------
    # Measurement
    # -------------------------------------------------------------------------
    
    def measure(self, target: int, deterministic: Optional[float] = None) -> int:
        """
        Measure target qubit, collapsing superposition.
        
        Args:
            target: Qubit index to measure
            deterministic: Optional float 0.0-1.0 for reproducible results (testing).
                          If None, uses true random from random.random()
                          
        Returns:
            0 or 1 measurement outcome
        """
        self._validate_qubit(target)
        
        rand_val = deterministic if deterministic is not None else random.random()
        result = C_LIB.measure_qubit(self.sys_ptr, target, ctypes.c_double(rand_val))
        
        self._measurement_history.append((target, result))
        self.circuit_log.append(f"MEASURE({target})->{result}")
        return result

    # -------------------------------------------------------------------------
    # State Inspection & Visualization
    # -------------------------------------------------------------------------
    
    def get_state_vector_probs(self) -> List[float]:
        """
        Extract full probability distribution from state vector.
        
        Returns:
            List of probabilities for each basis state (|000>, |001>, etc.)
        """
        size = self.sys_ptr.contents.state_size
        probs = []
        for i in range(size):
            p = C_LIB.get_probability(self.sys_ptr, i)
            probs.append(p)
        return probs

    def get_amplitudes(self) -> List[complex]:
        """
        Extract complex amplitudes (not just probabilities) from state vector.
        Note: This accesses the raw state_vector pointer directly.
        """
        size = self.sys_ptr.contents.state_size
        raw_vec = self.sys_ptr.contents.state_vector
        amps = []
        for i in range(size):
            c_val = raw_vec[i]
            amps.append(complex(c_val.real, c_val.imag))
        return amps

    def print_state(self, threshold: float = 0.001) -> None:
        """
        Pretty-print basis states with non-zero probability.
        
        Args:
            threshold: Minimum probability to display (filters noise/rounding errors)
        """
        print(f"\n{'─' * 50}")
        print(f"Quantum State | {self.num_qubits} Qubits | {2**self.num_qubits} basis states")
        print(f"{'─' * 50}")
        
        size = self.sys_ptr.contents.state_size
        raw_vec = self.sys_ptr.contents.state_vector
        
        found_nonzero = False
        for i in range(size):
            p = C_LIB.get_probability(self.sys_ptr, i)
            if p > threshold:
                found_nonzero = True
                c_val = raw_vec[i]
                amp = complex(c_val.real, c_val.imag)
                b_str = format(i, f'0{self.num_qubits}b')
                # Format amplitude nicely
                if abs(amp.imag) < 0.0001:
                    amp_str = f"{amp.real:+.3f}"
                else:
                    amp_str = f"{amp.real:+.3f}{amp.imag:+.3f}j"
                print(f"  |{b_str}⟩  {amp_str}  (p={p:.4f})")
        
        if not found_nonzero:
            print("  [Zero state or all probabilities below threshold]")
        print(f"{'─' * 50}")

    def plot_histogram(self, title: str = "Measurement Probabilities") -> None:
        """
        Display bar chart of state probabilities using matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Cannot plot histogram.")
            return
            
        probs = self.get_state_vector_probs()
        size = len(probs)
        labels = [format(i, f'0{self.num_qubits}b') for i in range(size)]
        
        # Filter very small probabilities for cleaner display
        active_data = [(l, p) for l, p in zip(labels, probs) if p > 0.0001]
        
        if not active_data:
            print("No significant probabilities to plot")
            return
            
        active_labels, active_probs = zip(*active_data)
        
        plt.figure(figsize=(max(6, len(active_labels) * 0.8), 5))
        plt.bar(active_labels, active_probs, color='teal', edgecolor='black', alpha=0.8)
        plt.ylabel("Probability")
        plt.xlabel("Basis State")
        plt.title(title)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Rotate labels if many qubits
        if self.num_qubits > 3:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        plt.show()

    def get_circuit_log(self) -> str:
        """Return formatted string of operations applied in sequence."""
        return "\n".join([f"{i+1:3d}: {op}" for i, op in enumerate(self.circuit_log)])


# ==============================================================================
# DEMONSTRATION FUNCTIONS
# ==============================================================================

def demo_bell_state():
    """
    Create Bell State |Φ+⟩ = (|00⟩ + |11⟩)/√2
    Demonstrates entanglement and correlation.
    """
    print("\n" + "=" * 60)
    print("DEMO: Bell State (|Φ+⟩) Creation & Measurement")
    print("=" * 60)
    
    sim = PhonexQ(num_qubits=2)
    
    # Circuit: H(0) -> CNOT(0,1)
    print("\nApplying Hadamard to Qubit 0...")
    sim.h(0)
    
    print("Applying CNOT (Control: Q0, Target: Q1)...")
    sim.cnot(0, 1)
    
    print("\nCircuit:")
    print(sim.get_circuit_log())
    
    # Display state
    sim.print_state()
    sim.plot_histogram("Bell State |Φ+⟩")
    
    # Demonstrate correlation
    print("\n--- Correlation Test ---")
    print("Measuring both qubits (should be identical due to entanglement):")
    m0 = sim.measure(0)
    m1 = sim.measure(1)
    print(f"Qubit 0: {m0}")
    print(f"Qubit 1: {m1}")
    print(f"Correlation: {'✓ MAINTAINED' if m0 == m1 else '✗ BROKEN (ERROR)'}")


def demo_grover_search():
    """
    Grover's Algorithm: Search for |11⟩ in 2-qubit database.
    Should amplify |11⟩ to ~100% probability after 1 iteration.
    """
    print("\n" + "=" * 60)
    print("DEMO: Grover's Search Algorithm (Target: |11⟩)")
    print("=" * 60)
    
    sim = PhonexQ(num_qubits=2)
    
    # Step 1: Initialize superposition
    print("\nStep 1: Creating uniform superposition...")
    sim.h(0).h(1)
    sim.print_state()
    
    # Step 2: Oracle (marks |11⟩ with negative phase using CZ)
    print("\nStep 2: Applying Oracle (marks target |11⟩)...")
    sim.cz(0, 1)  # Controlled-Z marks |11⟩
    sim.print_state()
    
    # Step 3: Diffusion/Amplification operator
    print("\nStep 3: Applying Diffusion Operator...")
    sim.h(0).h(1)
    sim.x(0).x(1)
    sim.h(1)
    sim.cnot(0, 1)
    sim.h(1)
    sim.x(0).x(1)
    sim.h(0).h(1)
    
    print("\nFinal State (should be ~100% |11⟩):")
    sim.print_state()
    sim.plot_histogram("Grover Search: Target |11⟩")
    
    # Verify with measurement
    result = sim.measure(0) * 2 + sim.measure(1)  # Convert to decimal
    print(f"\nMeasured state: {format(result, '02b')}")


def demo_quantum_teleportation():
    """
    Quantum Teleportation Protocol:
    Transfers unknown state from Alice to Bob using entanglement and classical communication.
    """
    print("\n" + "=" * 60)
    print("DEMO: Quantum Teleportation Protocol")
    print("=" * 60)
    
    sim = PhonexQ(num_qubits=3)
    
    # Qubit 0: State to teleport (unknown |ψ⟩)
    # We'll prepare it as |1⟩ for clarity (apply X)
    print("\nPreparing state to teleport (|1⟩ on Qubit 0)...")
    sim.x(0)
    
    # Step 1: Create entangled pair between Alice (Q1) and Bob (Q2)
    print("Creating entangled pair (Bell state on Q1, Q2)...")
    sim.h(1).cnot(1, 2)
    
    print("\nInitial combined state:")
    sim.print_state()
    
    # Step 2: Bell measurement (Alice's side)
    print("\nStep 2: Alice performs Bell measurement...")
    sim.cnot(0, 1)
    sim.h(0)
    
    # Measure Alice's qubits
    m0 = sim.measure(0)
    m1 = sim.measure(1)
    print(f"Classical bits sent to Bob: {m1}{m0}")
    
    # Step 3: Bob's corrections (controlled by classical bits)
    print("\nStep 3: Bob applies corrections...")
    if m1:
        sim.z(2)
    if m0:
        sim.x(2)
        
    print("Final state (Qubit 2 should be |1⟩):")
    sim.print_state()
    
    # Verify
    final = sim.measure(2)
    print(f"\nVerified: Teleported state is |{final}⟩ " + 
          ("✓ SUCCESS" if final == 1 else "✗ FAILURE"))


def demo_deutsch_jozsa():
    """
    Deutsch-Jozsa Algorithm: Determine if function is constant or balanced.
    Uses 2 qubits to test a promise problem.
    """
    print("\n" + "=" * 60)
    print("DEMO: Deutsch-Jozsa Algorithm (Balanced Function)")
    print("=" * 60)
    
    sim = PhonexQ(num_qubits=2)
    
    # Initialize: |01⟩
    sim.x(1)
    sim.h(0).h(1)
    
    # Oracle for balanced function f(x) = x (CNOT implements this)
    print("\nApplying Oracle (balanced function)...")
    sim.cnot(0, 1)
    
    sim.h(0)
    
    print("\nMeasuring Qubit 0:")
    result = sim.measure(0)
    print(f"Result: {result}")
    print(f"Function is: {'BALANCED' if result == 1 else 'CONSTANT'}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("PhonexQ Quantum Simulator")
    print("C-Backend Accelerated Quantum Computation")
    
    try:
        # Run all demonstrations
        demo_bell_state()
        demo_grover_search()
        demo_quantum_teleportation()
        demo_deutsch_jozsa()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)