#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

typedef double complex dcomplex;

typedef struct {
    unsigned int num_qubits;
    size_t state_size;
    dcomplex *state_vector;
} QuantumSystem;

QuantumSystem* init_system(unsigned int num_qubits) {
    // Prevent shift overflow (UB if >= 64) and unreasonable allocations
    if (num_qubits >= 64) {
        fprintf(stderr, "Error: num_qubits must be < 64\n");
        return NULL;
    }
    if (num_qubits == 0) {
        fprintf(stderr, "Error: num_qubits must be > 0\n");
        return NULL;
    }
    
    QuantumSystem* sys = malloc(sizeof(QuantumSystem));
    if (!sys) {
        perror("malloc failed");
        return NULL;
    }
    
    sys->num_qubits = num_qubits;
    sys->state_size = 1ULL << num_qubits;
    
    // Check for overflow (if size is 0 after shift, something went wrong)
    if (sys->state_size == 0) {
        free(sys);
        return NULL;
    }
    
    sys->state_vector = calloc(sys->state_size, sizeof(dcomplex));
    if (!sys->state_vector) {
        perror("calloc failed - state too large?");
        free(sys);
        return NULL;
    }
    
    sys->state_vector[0] = 1.0 + 0.0 * I;
    return sys;
}

void free_system(QuantumSystem* sys) {
    if (sys) {
        free(sys->state_vector);
        free(sys);
    }
}

static int check_system(const QuantumSystem* sys) {
    if (!sys || !sys->state_vector) {
        fprintf(stderr, "Error: NULL system\n");
        return 0;
    }
    return 1;
}

void apply_gate(QuantumSystem* sys, unsigned int target_qubit, 
                dcomplex u00, dcomplex u01, dcomplex u10, dcomplex u11) {
    if (!check_system(sys)) return;
    if (target_qubit >= sys->num_qubits) {
        fprintf(stderr, "Error: target_qubit %u out of bounds (max %u)\n", 
                target_qubit, sys->num_qubits - 1);
        return;
    }
    
    size_t size = sys->state_size;
    size_t bit = 1ULL << target_qubit;

    for (size_t i = 0; i < size; i += 2 * bit) {
        for (size_t j = i; j < i + bit; j++) {
            size_t idx0 = j;
            size_t idx1 = j | bit;

            dcomplex a = sys->state_vector[idx0];
            dcomplex b = sys->state_vector[idx1];

            sys->state_vector[idx0] = u00 * a + u01 * b;
            sys->state_vector[idx1] = u10 * a + u11 * b;
        }
    }
}

void apply_controlled_gate(QuantumSystem* sys, unsigned int control_qubit, 
                           unsigned int target_qubit,
                           dcomplex u00, dcomplex u01, dcomplex u10, dcomplex u11) {
    if (!check_system(sys)) return;
    if (target_qubit >= sys->num_qubits || control_qubit >= sys->num_qubits) {
        fprintf(stderr, "Error: qubit index out of bounds\n");
        return;
    }
    if (control_qubit == target_qubit) {
        fprintf(stderr, "Error: control and target must be different\n");
        return;
    }
    
    size_t size = sys->state_size;
    size_t t_bit = 1ULL << target_qubit;
    size_t c_bit = 1ULL << control_qubit;

    for (size_t i = 0; i < size; i += 2 * t_bit) {
        for (size_t j = i; j < i + t_bit; j++) {
            // Only apply if control bit is set in this pair
            if ((j & c_bit) != 0) {
                size_t idx0 = j;
                size_t idx1 = j | t_bit;

                dcomplex a = sys->state_vector[idx0];
                dcomplex b = sys->state_vector[idx1];

                sys->state_vector[idx0] = u00 * a + u01 * b;
                sys->state_vector[idx1] = u10 * a + u11 * b;
            }
        }
    }
}

int measure_qubit(QuantumSystem* sys, unsigned int target_qubit, double random_val) {
    if (!check_system(sys)) return -1;
    if (target_qubit >= sys->num_qubits) {
        fprintf(stderr, "Error: target_qubit out of bounds\n");
        return -1;
    }
    if (random_val < 0.0 || random_val > 1.0) {
        fprintf(stderr, "Error: random_val must be in [0,1]\n");
        return -1;
    }
    
    size_t size = sys->state_size;
    size_t mask = 1ULL << target_qubit;
    double prob_zero = 0.0;

    // Calculate P(0) - use cabs squared manually for efficiency
    for (size_t i = 0; i < size; i++) {
        if ((i & mask) == 0) {
            dcomplex amp = sys->state_vector[i];
            prob_zero += creal(amp) * creal(amp) + cimag(amp) * cimag(amp);
        }
    }

    // Handle edge cases to prevent division by zero
    if (prob_zero <= 0.0) {
        prob_zero = 0.0;
    } else if (prob_zero >= 1.0) {
        prob_zero = 1.0;
    }

    int result = (random_val < prob_zero) ? 0 : 1;
    double prob_result = (result == 0) ? prob_zero : (1.0 - prob_zero);
    
    if (prob_result < 1e-15) {
        fprintf(stderr, "Error: Impossible measurement result (prob ~0)\n");
        return -1;
    }
    
    double norm_factor = 1.0 / sqrt(prob_result);

    // Collapse and Normalize
    for (size_t i = 0; i < size; i++) {
        int bit_val = (i & mask) ? 1 : 0;
        if (bit_val == result) {
            sys->state_vector[i] *= norm_factor;
        } else {
            sys->state_vector[i] = 0.0 + 0.0 * I;
        }
    }
    return result;
}

double get_probability(QuantumSystem* sys, size_t state_index) {
    if (!check_system(sys)) return 0.0;
    if (state_index >= sys->state_size) return 0.0;
    dcomplex amp = sys->state_vector[state_index];
    return creal(amp) * creal(amp) + cimag(amp) * cimag(amp);
}