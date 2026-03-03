# chemoton_sim
#!/usr/bin/env python3
"""
Optimized Chemoton Evolution Simulator
===========================
Based on the original Chemoton simulator with improved performance and stability:

Enhanced features:
- SciPy's ODE solvers for improved stability and speed
- Numba acceleration for core computations
- Adaptive time stepping for better convergence
- Vectorized operations for population simulations
- Improved error handling and simulation stability
- Memory optimizations for large populations

Usage:
python optimized_chemoton.py --generations 50 --solver lsoda --plot-stats
"""


from __future__ import annotations
import csv
import numpy as np
import argparse
import pathlib
import datetime as _dt
import sys
import copy
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Callable
from scipy.integrate import solve_ivp
import numba as nb
from tqdm import tqdm
import random


# ---------------------------------------------------------------------------
# Base Parameters ----------------------------------------------------------
# ---------------------------------------------------------------------------

BASE_PAR = {
    # forward rates ---------------------------------------------------------
    'k1':  2.0,  'k2': 50.0, 'k3': 50.0, 'k4': 50.0, 'k5': 10.0,
    'k6':  100.0, 'k7': 100.0,  'k8': 10.0,  'k9': 10.0, 'k10':10.0,
    # reverse rates ---------------------------------------------------------
    'k1p': 0.1,  'k2p': 0.1,   'k3p': 0.1,  'k4p': 0.1, 'k5p': 0.1,
    'k6p': 1.0,  'k9p': 0.1,
    # thresholds & geometry -------------------------------------------------
    'V_th': 35.0,     # monomer threshold for strand separation
    'N'   : 25,       # template length (monomers per strand)
    'kS' : 0.5,       #rate of S consumption when food is gone
    'kSp' : 0.1,      # reverse rate
    'V_th_2' : 50.0,   # the value to know there is enough V for now and can stat storing
    'k_store' : 0.5,
    'k_release' : 1.0,
    'k_leak' : 0.1,
}

no_food = 1.0

# ---------------------------------------------------------------------------
# Dynamic index table -------------------------------------------------------
# ---------------------------------------------------------------------------

BASE_NAMES = [
    'A1','A2','A3','A4','A5',    # metabolism A
    'B1','B2','B3','B4','B5',    # metabolism B
    "V'", 'V_store', 'R', "T'", 'T*', 'T',  # monomer + residues + membr.
    'S','Q',                     # surface & volume proxy
    'tmpl_len',                  # diagnostic: ⟨strand length⟩ (fdp only)
    'has_met_B',                 # indicator for metabolism B presence
    'X',
    'Z',
]
IDX: dict[str,int] = {n:i for i,n in enumerate(BASE_NAMES)}

def _add_idx(name:str) -> int:
    """Add a new index to the global index dictionary."""
    if name not in IDX:
        IDX[name] = len(IDX)
    return IDX[name]

# Pre-compute common indices for performance
A1_IDX, A2_IDX, A3_IDX, A4_IDX, A5_IDX = [IDX[f'A{i}'] for i in range(1,6)]
B1_IDX, B2_IDX, B3_IDX, B4_IDX, B5_IDX = [IDX[f'B{i}'] for i in range(1,6)]
V_IDX, R_IDX = IDX["V'"], IDX['R']
TP_IDX, TS_IDX, T_IDX = IDX["T'"], IDX['T*'], IDX['T']
S_IDX, Q_IDX = IDX['S'], IDX['Q']
TMPL_LEN_IDX, HAS_MET_B_IDX = IDX['tmpl_len'], IDX['has_met_B']
X_IDX = IDX['X']
Z_IDX = IDX['Z']
VSTORE_IDX = IDX['V_store']


# ---------------------------------------------------------------------------
# Optimized Template kinetics -----------------------------------------------
# ---------------------------------------------------------------------------

Vec = np.ndarray

@nb.njit

def _ganti_rhs_numba(y: np.ndarray, V_th: float, k7: float, N: int, pV0_idx: int) -> Tuple[float, float]:
    """Single‑slot template replication - optimized with Numba."""
    V = y[V_IDX]
    if V <= V_th:
        return 0.0, 0.0
    rate = k7 * y[pV0_idx] * V
    dT = rate                     # templates doubled
    dV = -(N/2) * rate            # monomers consumed
    return dT, dV

#------------------------------------------------------

@nb.njit
def _fdp_rhs_numba(y: np.ndarray, V_th: float, k6: float, k6p: float, k7: float, N: int, 
                  pV_indices: np.ndarray, R_idx: int) -> Tuple[np.ndarray, float]:
    """Multi‑stage replication (Fernando & Di Paolo) - optimized with Numba."""
    V = y[V_IDX]
    dtmp = np.zeros(N)
    dV = 0.0

    # -- initiation (Eqn.6) -------------------------------------------------
    if V > V_th:
        rate_i = k6 * y[pV_indices[0]] * V
        rate_i_rev = k6p * y[pV_indices[1]] * y[R_idx]
        # pV0 → pV1
        dtmp[0] -= rate_i
        dtmp[1] += rate_i - rate_i_rev
        # monomer balance (Eqn.9)
        dV -= rate_i * (N/2)
        dV += rate_i_rev

    # -- elongation (Eqn.8) -------------------------------------------------
    for r in range(1, N-1):
        src = pV_indices[r]
        rate = k7 * y[src] * V
        dtmp[r] -= rate
        dtmp[r+1] += rate
        # monomer consumption termsz
        dV -= rate * (N/2)/(N-1)

    # -- termination (strand separation) (Eqn.7) ----------------------------
    rate_t = k7 * y[pV_indices[N-1]] * V
    dtmp[N-1] -= rate_t
    dtmp[0] += 2*rate_t
    dV -= rate_t * (N/2)/(N-1)

    return dtmp, dV

# ---------------------------------------------------------------------------
# Core RHS (optimized) ------------------------------------------------------
# ---------------------------------------------------------------------------

def make_pv_indices(N: int, template: str) -> np.ndarray:
    """Create array of pV indices for efficient access."""
    if template == 'fdp':
        return np.array([IDX[f'pV{r}'] for r in range(N)])
    else:  # ganti
        return np.array([IDX['pV0']])

def create_parameter_array(parameters: dict) -> np.ndarray:
    """Convert parameter dictionary to array for efficient access in Numba."""
    param_names = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10',
                   'k1p', 'k2p', 'k3p', 'k4p', 'k5p', 'k6p', 'k9p',
                   'V_th', 'N', 'kS','kSp', 'k_release', 'k_store', 'V_th_2', 'k_leak']
    return np.array([parameters[name] for name in param_names])

#@nb.njit
def rhs_core_numba(t: float, y: np.ndarray, param_array: np.ndarray, 
                   pv_indices: np.ndarray, is_fdp: bool) -> np.ndarray:
    """Core RHS function optimized with Numba."""
    # Extract parameters from array
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = param_array[:10]
    k1p, k2p, k3p, k4p, k5p, k6p, k9p = param_array[10:17]
    V_th, N = param_array[17:19]
    kS, kSp, k_release, k_store, V_th_2, k_leak = param_array[19:25]
    N = int(N)  # Ensure N is integer
    
    # Clip values to avoid numerical issues
    y_safe = np.clip(y, 0, 1e10)
    
    X = y_safe[X_IDX]      # add this line
    Z = y_safe[Z_IDX]      # add if you use Z

    # Initialize derivatives
    dy = np.zeros_like(y)
    

#------fat storing mechanism-------------------------------------------

    V = y_safe[V_IDX]
    store_rate = 0.0
    V_store = y_safe[VSTORE_IDX]

    #no_food = 1
    hunger = (X <= no_food)

    if V > V_th_2:
        store_rate = k_store * (V - V_th_2)   # only store the excess

    dy[V_IDX] -= store_rate
    dy[VSTORE_IDX] += store_rate
    
   
# unavoidable dissipation of stored energy:
    dy[VSTORE_IDX] -= k_leak * V_store

#release script when we need fat to use:
    if hunger and V < V_th and V_store > 1.0:
        release = k_release * min(V_store, V_th - V)
        dy[VSTORE_IDX] -= release
        dy[V_IDX] += release


    # ------------ metabolism A ----------------------------------------------
    A1, A2, A3, A4, A5 = y_safe[A1_IDX], y_safe[A2_IDX], y_safe[A3_IDX], y_safe[A4_IDX], y_safe[A5_IDX]
    R, S = y_safe[R_IDX], y_safe[S_IDX]
    TP, TS, T = y_safe[TP_IDX], y_safe[TS_IDX], y_safe[T_IDX]
    

    # Metabolism A differential equations
    dy[A1_IDX] = 2*(k5*A5 - k5p*A1*A1) - k1*A1*X + k1p*A2
    dy[A2_IDX] = k1*A1*X - k1p*A2 - k2*A2 + k2p*A3*0.1
    dy[A3_IDX] = k2*A2 - k2p*A3*0.1 - k3*A3 + k3p*A4*V
    dy[A4_IDX] = k3*A3 - k3p*A4*V - k4*A4 + k4p*A5*TP
    dy[A5_IDX] = k4*A4 - k4p*A5*TP - k5*A5 + k5p*A1*A1


# --- hunger behaviour: use S as fuel when X is tiny ---

# baseline membrane dynamics
    if hunger:
        dS = 0.0
    else:
        dS = k10 * T * S

    if hunger and S > 1 and V < V_th and V_store < 1:
    # consume S through an extra A1<->A2 pathway
        dS -= kS * A1 * S
        dy[A1_IDX] += -kS * A1 * S + kSp * A2
        dy[A2_IDX] +=  kS * A1 * S - kSp * A2

    dy[S_IDX] = dS
    dy[Q_IDX] = dS
    
    # ------------ metabolism B (if present) --------------------------------
    has_met_B = y_safe[HAS_MET_B_IDX] > 0.5
    if has_met_B:
        B1, B2, B3, B4, B5 = (y_safe[B1_IDX], y_safe[B2_IDX], y_safe[B3_IDX], 
                              y_safe[B4_IDX], y_safe[B5_IDX])

        # Metabolism B differential equations
        dy[B1_IDX] = 2*(k5*B5 - k5p*B1*B1) - k1*B1*Z + k1p*B2
        dy[B2_IDX] = k1*B1*Z - k1p*B2 - k2*B2 + k2p*B3*0.1
        dy[B3_IDX] = k2*B2 - k2p*B3*0.1 - k3*B3 + k3p*B4*V
        dy[B4_IDX] = k3*B3 - k3p*B4*V - k4*B4 + k4p*B5*TP
        dy[B5_IDX] = k4*B4 - k4p*B5*TP - k5*B5 + k5p*B1*B1
        
    hunger_B = (X <= no_food) and (Z <= no_food)

    if has_met_B and hunger_B and S > 1 and V < V_th and V_store < 1:
        dS -= kS * B1 * S
        dy[B1_IDX] += -kS * B1 * S + kSp * B2
        dy[B2_IDX] +=  kS * B1 * S - kSp * B2


    # ------------ template system -------------------------------------------
    if not is_fdp:  # ganti mode
        dT, dV_tmpl = _ganti_rhs_numba(y_safe, V_th, k7, N, pv_indices[0])
        dy[pv_indices[0]] += dT
        dy[V_IDX] += dV_tmpl
    else:  # fdp mode
        dtmp, dV_tmpl = _fdp_rhs_numba(y_safe, V_th, k6, k6p, k7, N, pv_indices, R_IDX)
        for r in range(N):
            dy[pv_indices[r]] += dtmp[r]
        dy[V_IDX] += dV_tmpl
        
        # smooth mean strand length
        total = 0.0
        avg = 0.0
        for r in range(N):
            total += y_safe[pv_indices[r]]
        
        if total > 1e-12:
            for r in range(N):
                avg += (N-r) * y_safe[pv_indices[r]]
            avg /= total
            dy[TMPL_LEN_IDX] = 0.2 * (avg - y_safe[TMPL_LEN_IDX])
    
    # ------------ monomer & residue dynamics -------------------------------
    # 1) metabolism A-derived monomer
    dV_met_A = k3*A3 - k3p*A4*V
    
    # 2) metabolism B-derived monomer (if present)
    dV_met_B = 0.0
    if has_met_B:
        dV_met_B = k3*y_safe[B3_IDX] - k3p*y_safe[B4_IDX]*V
    
    # 3) initiation reverse reaction
    dV_rev_i = 0.0
    if is_fdp:
        dV_rev_i = k6p * y_safe[pv_indices[1]] * R
    
    # 4) initiation forward consumption
    dV_i = 0.0
    if is_fdp:
        dV_i = -k6 * y_safe[pv_indices[0]] * V
    
    # 5) elongation consumption
    dV_el = 0.0
    if is_fdp:
        for r in range(1, N):
            dV_el -= k7 * y_safe[pv_indices[r]] * V
    else:
        dV_el = -k7 * y_safe[pv_indices[0]] * V
    
    dy[V_IDX] += dV_met_A + dV_met_B + dV_rev_i + dV_i + dV_el
    
    # R dynamics
    dR_i = 0.0
    dR_rev_i = 0.0
    if is_fdp:
        dR_i = k6 * y_safe[pv_indices[0]] * V
        dR_rev_i = -k6p * y_safe[pv_indices[1]] * R
    
    dR_mem = +k9p*T - k9*TS*R
    
    dR_el = 0.0
    if is_fdp:
        for r in range(1, N):
            dR_el += k7 * y_safe[pv_indices[r]] * V
    else:
        dR_el = k7 * y_safe[pv_indices[0]] * V
    
    dy[R_IDX] = dR_i + dR_rev_i + dR_mem + dR_el
    
    # ------------ membrane --------------------------------------------------
    dy[TP_IDX] = k4*A4 - k4p*A5*TP - k8*TP
    if has_met_B:
        dy[TP_IDX] += k4*y_safe[B4_IDX] - k4p*y_safe[B5_IDX]*TP
    
    dy[TS_IDX] = k8*TP - k9*TS*R + k9p*T
    dy[T_IDX] = k9*TS*R - k9p*T - k10*T*S
    
    # ------------ geometry --------------------------------------------------

    return dy

def rhs_wrapper(t: float, y: np.ndarray, param_array: np.ndarray,
                pv_indices: np.ndarray, is_fdp: bool) -> np.ndarray:
    try:
        out = rhs_core_numba(t, y, param_array, pv_indices, is_fdp)
    except Exception as e:
        print(f"\nError in RHS at t={t}: {e}")
        raise  # ← important: stop immediately and show a traceback

    # sanity: solver must get a 1D vector with same length as y
    out = np.asarray(out, dtype=float)
    if out.ndim != 1 or out.size != y.size:
        raise ValueError(f"RHS size mismatch: got {out.shape}, expected {(y.size,)}")
    return out

# ---------------------------------------------------------------------------
# ODE Solvers ---------------------------------------------------------------
# ---------------------------------------------------------------------------


def solve_with_scipy(y0: np.ndarray, t_span: Tuple[float, float], 
                     param_array: np.ndarray, pv_indices: np.ndarray, 
                     is_fdp: bool, method: str = 'LSODA', 
                     rtol: float = 1e-4, atol: float = 1e-6) -> np.ndarray:
    """Solve ODE system using SciPy's ODE solvers."""
    def f(t, y):
        return rhs_wrapper(t, y, param_array, pv_indices, is_fdp)
    
    # Solve the system
    sol = solve_ivp(
        f, t_span, y0, 
        method=method,
        rtol=rtol, atol=atol,
        dense_output=True  # Enable dense output for flexible time sampling
    )
    
    if not sol.success:
        print(f"Warning: Solver failed: {sol.message}")
        return y0  # Return initial state if solver fails
    
    return sol.y[:, -1]  # Return final state

#----------------------------------------------------------------

def step_euler(y: np.ndarray, dt: float, param_array: np.ndarray, 
               pv_indices: np.ndarray, is_fdp: bool) -> np.ndarray:
    """Euler method for ODE integration (optimized)."""
    dydt = rhs_core_numba(0.0, y.copy(), param_array, pv_indices, is_fdp)
    new_y = y + dt * dydt
    return np.clip(new_y, 0, 1e10)  # Prevent negative values

#----------------------------------------------------------------

def step_rk4(y: np.ndarray, dt: float, param_array: np.ndarray, 
             pv_indices: np.ndarray, is_fdp: bool) -> np.ndarray:
    """RK4 method for ODE integration (optimized)."""
    k1 = rhs_core_numba(0.0, y.copy(), param_array, pv_indices, is_fdp)
    k2 = rhs_core_numba(dt/2, y + 0.5*dt*k1, param_array, pv_indices, is_fdp)
    k3 = rhs_core_numba(dt/2, y + 0.5*dt*k2, param_array, pv_indices, is_fdp)
    k4 = rhs_core_numba(dt, y + dt*k3, param_array, pv_indices, is_fdp)
    
    new_y = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return np.clip(new_y, 0, 1e10)  # Prevent negative values

STEP = {
    'euler': step_euler,
    'rk4': step_rk4, 
    'lsoda': lambda y, dt, param_array, pv_indices, is_fdp: 
        solve_with_scipy(y, (0, dt), param_array, pv_indices, is_fdp, 'LSODA'),
    'radau': lambda y, dt, param_array, pv_indices, is_fdp: 
        solve_with_scipy(y, (0, dt), param_array, pv_indices, is_fdp, 'Radau'),
    'bdf': lambda y, dt, param_array, pv_indices, is_fdp: 
        solve_with_scipy(y, (0, dt), param_array, pv_indices, is_fdp, 'BDF'),
    'dopri5': lambda y, dt, param_array, pv_indices, is_fdp: 
        solve_with_scipy(y, (0, dt), param_array, pv_indices, is_fdp, 'DOP853')
}

# ---------------------------------------------------------------------------
# Chemoton Class (optimized) ------------------------------------------------
# ---------------------------------------------------------------------------

class Chemoton:
    """A single Chemoton cell with its own parameters and state (optimized)"""
   
#----------------------------------------------------------------
 
    def __init__(self, parameters=None, template='ganti', x=0, y=0):
        self.parameters = parameters or copy.deepcopy(BASE_PAR)
        self.template = template
        self.divisions = 0
        self.alive = True
        self.death_cause = None
        self.cell_dead = False
        self.x = x
        self.y = y
        self.want_move = False
        self.move_dx = 0
        self.move_dy = 0


        # For efficient computation
        self.param_array = None
        self.pv_indices = None
        self.is_fdp = template == 'fdp'
        
        self.initialize_state()
   
#----------------------------------------------------------------
 
    def _update_computation_arrays(self):
        """Update arrays used for efficient computation."""
        self.param_array = create_parameter_array(self.parameters)
        self.pv_indices = make_pv_indices(int(self.parameters['N']), self.template)
  
#----------------------------------------------------------------
      
    def initialize_state(self):
        """Initialize the state vector according to the template model"""
        N = int(self.parameters['N'])
        
        # Expand state vector for template model
        if self.template == 'fdp':
            for r in range(N):
                _add_idx(f'pV{r}')
        else:
            _add_idx('pV0')
        
        # Initialize state vector
        self.state = np.zeros(len(IDX))
        
        # Set initial values
        self.state[A1_IDX:A5_IDX+1] = [1, 1.8, 1.9, 1.7, 10]
        self.state[V_IDX] = 40.0  # exceed V_th to trigger first replication
        self.state[IDX.get('pV0', -1)] = 0.01  # seed one template strand
        self.state[R_IDX] = 0.5  # residue pool seed
        self.state[TP_IDX], self.state[TS_IDX], self.state[T_IDX] = 17, 14, 0
        self.state[S_IDX] = 1.5
        self.state[Q_IDX] = 1.0
        self.state[TMPL_LEN_IDX] = N
        
        # Initialize metabolism B if N ≥ 40
        if N >= 40:
            self.activate_metabolism_B()
        else:
            self.state[HAS_MET_B_IDX] = 0  # metabolism B inactive
            # Initialize B metabolites to zero
            self.state[B1_IDX:B5_IDX+1] = [0, 0, 0, 0, 0]
        
        # Initialize computation arrays
        self._update_computation_arrays()
        


#----------------------------------------------------------------

    def activate_metabolism_B(self):
        """Activate metabolism B in this Chemoton"""
        self.state[HAS_MET_B_IDX] = 1  # Set flag for metabolism B
        self.state[B1_IDX:B5_IDX+1] = [0.8, 1.5, 1.6, 1.4, 8]  # Initial values for B metabolites
    

#----------------------------------------------------------------

    def simulate_step(self, dt, method='rk4'):
        """Simulate one time step with improved stability"""
        if not self.alive:
            return False
        
        # Check if N is below survival threshold
        if self.parameters['N'] < 20:
            self.death_cause = "N_too_small"
            self.alive = False
            return False
        
        # Select stepper function
        if method not in STEP:
            print(f"Warning: Unknown solver '{method}', falling back to RK4")
            method = 'rk4'
        
        step = STEP[method]
        
        # Implement adaptive time stepping for better stability
        try:
            # Use smaller time steps for better stability
            num_substeps = 1
            if method in ['euler', 'rk4']:
                num_substeps = max(1, int(dt / 1e-5))  # Subdivide large steps
            
            sub_dt = dt / num_substeps
            
            # Perform integration
            old_state = self.state.copy()
            new_state = old_state.copy()
            
            for _ in range(num_substeps):
                new_state = step(new_state, sub_dt, self.param_array, self.pv_indices, self.is_fdp)
                
                # Check for NaN values during substeps
                if np.any(~np.isfinite(new_state)):
                    print("Warning: Non-finite values in substep, reducing step size")
                    sub_dt /= 2
                    new_state = old_state.copy()
                    if sub_dt < 1e-12:
                        return False  # Too small, abort
            
            self.state = new_state
            
            # Final NaN check
            if np.any(~np.isfinite(self.state)):
                print("Warning: Non-finite values in state, restoring previous state")
                self.state = old_state
                return False
            
            if self.state[S_IDX] <= 1:
                if self.state[X_IDX] <= no_food and self.state[Z_IDX] <= no_food:
                    self.death_cause = "hunger_shrink"
                else:
                    self.death_cause = "shrink_other"
                self.alive = False
                return False



        except Exception as e:
            print(f"Error during simulation step: {e}")
            return False
        
        # Check for division
        S0 = 1.0  # Initial surface area
        if self.state[S_IDX] >= 2 * S0:
            # Division occurs
            for i in range(len(self.state)):
                if i not in {S_IDX, Q_IDX, TMPL_LEN_IDX, HAS_MET_B_IDX}:
                    self.state[i] *= 0.5
            self.state[S_IDX] *= 0.5
            self.state[Q_IDX] *= 0.5
            S0 = self.state[S_IDX]
            self.divisions += 1
            return True  # Division occurred

        #movement rule:

        X = self.state[X_IDX]
        Z = self.state[Z_IDX]
        V_store = self.state[VSTORE_IDX]
        S = self.state[S_IDX]

        #if X <= no_food and Z <= no_food:
            #print("HUNGRY | V_store:", V_store, "S:", S)

        if X < no_food:
            self.want_move = True
            self.move_dx = random.choice([-1, 0, 1])
            self.move_dy = random.choice([-1, 0, 1])
        else:
            self.want_move = False
            self.move_dx = 0
            self.move_dy = 0

        return False



    
#----------------------------------------------------------------

    def mutate(self, mutation_std=5.0):
        """Apply mutation to template length N"""
        # Calculate new N value with Gaussian noise
        old_N = int(self.parameters['N'])
        new_N = max(1, round(old_N + np.random.normal(0, mutation_std)))
        
        # If N changes, update the parameter and reinitialize
        if new_N != old_N:
            self.parameters['N'] = new_N
            
            # Update param array
            self.param_array = create_parameter_array(self.parameters)
            
            # If template model requires different indices, reinitialize
            if self.template == 'fdp' and new_N > old_N:
                # Add any missing indices for the new N
                for r in range(old_N, new_N):
                    _add_idx(f'pV{r}')
                # Update state array size if needed
                if len(self.state) < len(IDX):
                    new_size = len(IDX)
                    new_state = np.zeros(new_size)
                    new_state[:len(self.state)] = self.state
                    self.state = new_state
            
            # Update PV indices
            self.pv_indices = make_pv_indices(new_N, self.template)
            
            # Check for potential gain of metabolism B if N ≥ 40
            if new_N >= 40 and self.state[HAS_MET_B_IDX] < 0.5:
                # 30% chance to gain metabolism B when N ≥ 40
                if np.random.random() < 0.3:
                    self.activate_metabolism_B()
            
            # Reset divisions counter
            self.divisions = 0
            
            # Refresh alive status
            if new_N < 20:
                self.death_cause = "N_too_small"
                self.alive = False

#self.alive = new_N >= 20
        
        return new_N



#part that links with environment --------------------------------------------------------

    def compute_consumption(self, X: float, Z: float) -> tuple[float, float]:
 
        k1 = self.parameters["k1"]

        # X consumption via metabolism A
        A1 = self.state[A1_IDX]
        cons_X = k1 * A1 * X

        # Z consumption via metabolism B (if present)
        cons_Z = 0.0
        if self.state[HAS_MET_B_IDX] > 0.5:
            B1 = self.state[B1_IDX]
            cons_Z = k1 * B1 * Z

        return cons_X, cons_Z



# ---------------------------------------------------------------------------
# Parallelized Population Simulator -----------------------------------------
# ---------------------------------------------------------------------------

class ChemotonPopulation:
    """Manage a population of Chemotons with performance optimizations"""
    

#----------------------------------------------------------------

    def __init__(self, population_size=10, template='ganti', initial_N_mean=25, initial_N_std=3):
        self.population_size = population_size
        self.template = template
        self.generation = 0
        self.total_individuals = 0
        

        # Create initial population with varied template lengths
        self.population = []
        for _ in range(population_size):
            parameters = copy.deepcopy(BASE_PAR)
            # Sample N from normal distribution around mean 25
            N = max(20, round(np.random.normal(initial_N_mean, initial_N_std)))
            parameters['N'] = N
            self.population.append(Chemoton(parameters, template))
    
#----------------------------------------------------------------

    def simulate_generation(self, target_population=100, dt=1e-4, method='rk4'):
        """Simulate one generation until target population is reached"""
        offspring = []  # Store new individuals                    

        # Run until we reach target population or all individuals are dead
        with tqdm(total=target_population-len(self.population), 
                  desc=f"Gen {self.generation} offspring", 
                  disable=False) as pbar:
            
            while len(offspring) + len(self.population) < target_population:
                if not any(chemoton.alive for chemoton in self.population):
                    print(f"Generation {self.generation} extinct! No viable individuals.")
                    break
                
                new_offspring_count = len(offspring)


#---------------------------------------------------------------

                # Simulate each individual step by step
                for chemoton in self.population:
                    if not chemoton.alive:
                        continue
                    
                    alive_before = chemoton.alive
                    # Simulate step and check for division
                    division_occurred = chemoton.simulate_step(dt, method)
                 

                    if division_occurred:
                        # Create offspring with same parameters as parent
                        child = Chemoton(copy.deepcopy(chemoton.parameters), self.template, x=chemoton.x, y=chemoton.y)
                        offspring.append(child)
                        
                        # Stop if we've reached target population
                        if len(offspring) + len(self.population) >= target_population:
                            break

                # Update progress bar
                pbar.update(len(offspring) - new_offspring_count)
        
        # Combine parent and offspring populations
        combined_population = self.population + offspring
        
        # Collect statistics before selection
        n_values = [chemoton.parameters['N'] for chemoton in combined_population if chemoton.alive]
        has_met_B = [chemoton.state[HAS_MET_B_IDX] > 0.5 for chemoton in combined_population if chemoton.alive]
        
        stats = {
            'generation': self.generation,
            'population_size': len(combined_population),
            'viable_count': len(n_values),
            'mean_N': np.mean(n_values) if n_values else 0,
            'min_N': min(n_values) if n_values else 0,
            'max_N': max(n_values) if n_values else 0,
            'percent_with_met_B': 100 * sum(has_met_B) / len(has_met_B) if has_met_B else 0,
            'simulation_time': dt * len(offspring),  # Approximate simulation time
        }
        
        # Update generation counter
        self.generation += 1
        self.total_individuals += len(offspring)
        
        return combined_population, stats

    
#----------------------------------------------------------------

    def select_next_generation(self, combined_population, mutation_std=5.0):
        """Randomly select individuals for next generation and apply mutations"""
        # Filter only alive individuals
        viable_population = [c for c in combined_population if c.alive]
        
        if len(viable_population) < self.population_size:
            print(f"Warning: Only {len(viable_population)} viable individuals available for selection")
        
        # Select random subset for next generation
        if viable_population:
            # Use weighted selection with slight preference for higher N values
            weights = np.array([c.parameters['N'] for c in viable_population])
            weights = weights / np.sum(weights)  # Normalize
            
            if len(viable_population) <= self.population_size:
                # Take all viable individuals if we don't have enough
                selected = viable_population
            else:
                # Weighted random selection without replacement
                indices = np.random.choice(
                    len(viable_population), 
                    size=self.population_size, 
                    replace=False,
                    p=weights
                )
                selected = [viable_population[i] for i in indices]
            
            self.population = selected
        else:
            # Population extinct, create new random individuals
            print("Population extinct! Creating new random individuals...")
            self.population = []
            for _ in range(self.population_size):
                parameters = copy.deepcopy(BASE_PAR)
                N = max(20, round(np.random.normal(25, 3)))  # Restart with viable N values
                parameters['N'] = N
                self.population.append(Chemoton(parameters, self.template))
        
        # Apply mutations to all individuals in new generation
        for chemoton in self.population:
            chemoton.mutate(mutation_std)
   
#----------------------------------------------------------------

    def run_evolution(self, generations=50, target_population=100, dt=1e-4, method='lsoda', mutation_std=5.0):
        """Run evolutionary simulation for specified number of generations"""
        all_stats = []
        
        print(f"Running evolution with {method} solver for {generations} generations")
        print(f"Initial population: {self.population_size}, Target: {target_population}")

#----------------------------------------------------------------


        for gen in range(generations):
            # Simulate generation
            combined_population, stats = self.simulate_generation(target_population, dt, method)
            
            v = [c.state[VSTORE_IDX] for c in combined_population if c.alive]
            print("Gen", stats["generation"], "VSTORE max", max(v) if v else 0.0)

            all_stats.append(stats)

                            
            # Display progress
            print(f"Generation {gen+1}/{generations} complete:")
            print(f"  Population: {stats['viable_count']}/{stats['population_size']} viable")
            print(f"  Template length N - Mean: {stats['mean_N']:.2f}, Min: {stats['min_N']}, Max: {stats['max_N']}")
            print(f"  Metabolism B present: {stats['percent_with_met_B']:.1f}%")

            # Select next generation
            self.select_next_generation(combined_population, mutation_std)
        

        return all_stats



# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def cli(argv=None):
    p = argparse.ArgumentParser(description='Optimized Chemoton Evolution Simulator')
    p.add_argument('--template', choices=['ganti', 'fdp'], default='ganti',
                   help='template replication model')
    p.add_argument('--generations', type=int, default=50, 
                   help='number of generations to simulate')
    p.add_argument('--population', type=int, default=10,
                   help='initial population size')
    p.add_argument('--target', type=int, default=100,
                   help='target population size per generation')
    p.add_argument('--dt', type=float, default=1e-4, 
                   help='time step for ODE integration')
    p.add_argument('--solver', choices=['euler', 'rk4', 'lsoda', 'radau', 'bdf', 'dopri5'], 
                   default='lsoda', help='ODE solver method')
    p.add_argument('--mutation-std', type=float, default=2.0,
                   help='standard deviation for mutations of N')
    
    # plotting arguments
    p.add_argument('--plot-stats', action='store_true',
                   help='plot evolution statistics')
    p.add_argument('--plot-final', action='store_true',
                   help='plot final population distribution')
    p.add_argument('--plot-performance', action='store_true',
                   help='plot solver performance statistics')
    p.add_argument('--noshow', action='store_true', 
                   help='suppress GUI display')
    p.add_argument('--save', action='store_true',
                   help='save figures to directory')
    p.add_argument('--plot-hunger', action='store_true',
                   help='plot deaths from hunger-driven shrinking per generation')
    p.add_argument('--plot-food', action='store_true',
                   help='plot food (X) across generations')
    
    # benchmarking
    p.add_argument('--benchmark', action='store_true',
                   help='run benchmarks comparing solvers')
    
    args = p.parse_args(argv)
    
    # Setup output directory if saving
    out = None
    if args.save:
        stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        out = pathlib.Path(f"optimized_chemoton_{stamp}")
    
    if args.benchmark:
        run_benchmarks(args, out)
        return
    
    # Initialize population
    population = ChemotonPopulation(
        population_size=args.population,
        template=args.template,
        initial_N_mean=25,
        initial_N_std=3
    )
    
    # Run evolution
    stats = population.run_evolution(
        generations=args.generations,
        target_population=args.target,
        dt=args.dt,
        method=args.solver,
        mutation_std=args.mutation_std
    )
    
    # Plot results
    if args.plot_stats or not (args.plot_final or args.plot_stats):
        plot_evolution_stats(stats, out, show=not args.noshow)
    
    if args.plot_final:
        plot_final_N_distribution(population.population, out, show=not args.noshow)
    
    if args.plot_performance:
        plot_performance_comparison(stats, args.solver, out, show=not args.noshow)

    if args.plot_hunger:
        plot_hunger(stats, out, show=not args.noshow)

    if args.plot_food:
        plot_food(stats, out, show=not args.noshow)
    
    # Print final statistics
    final_N = [chemoton.parameters['N'] for chemoton in population.population if chemoton.alive]
    final_met_B = [chemoton.state[HAS_MET_B_IDX] > 0.5 for chemoton in population.population if chemoton.alive]
    
    print("\nFinal Population Statistics:")
    print(f"  Viable individuals: {len(final_N)}/{args.population}")
    print(f"  Template length N - Mean: {np.mean(final_N):.2f}, Min: {min(final_N) if final_N else 0}, Max: {max(final_N) if final_N else 0}")
    print(f"  Metabolism B present: {100 * sum(final_met_B)/len(final_met_B) if final_met_B else 0:.1f}%")

def run_benchmarks(args, out=None):
    """Run benchmarks comparing different solvers"""
    solvers = ['euler', 'rk4', 'lsoda', 'bdf']
    results = {}
    
    for solver in solvers:
        print(f"\n--- Benchmarking {solver.upper()} solver ---")
        
        # Initialize population
        population = ChemotonPopulation(
            population_size=5,  # Small population for quick benchmarking
            template=args.template,
            initial_N_mean=25,
            initial_N_std=3
        )
        
        # Time the simulation
        start_time = _dt.datetime.now()
        
        stats = population.run_evolution(
            generations=5,  # Few generations for benchmarking
            target_population=20,  # Small target for quick benchmarking
            dt=args.dt,
            method=solver,
            mutation_std=args.mutation_std
        )
        
        end_time = _dt.datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"Completed in {elapsed:.2f} seconds")
        results[solver] = elapsed
    
    # Plot benchmark results
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    solvers_list = list(results.keys())
    times = [results[s] for s in solvers_list]
    
    bars = ax.bar(solvers_list, times, color='skyblue', edgecolor='navy')
    
    # Add time labels above each bar
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time:.2f}s', ha='center', va='bottom')
    
    ax.set_xlabel('Solver', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Performance Comparison of ODE Solvers', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure if requested
    if out:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "solver_benchmark.png", dpi=160, bbox_inches='tight')
    
    # Show figure if requested
    if not args.noshow:
        plt.show()
    
    plt.close(fig)
    
    # Print summary
    print("\nSolver Performance Summary:")
    for solver, time in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {solver.upper()}: {time:.2f} seconds")
    
    fastest = min(results.items(), key=lambda x: x[1])[0]
    print(f"\nFastest solver: {fastest.upper()}")

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        sys.exit(0)


