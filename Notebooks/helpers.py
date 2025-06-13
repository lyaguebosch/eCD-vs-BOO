#Package importation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

from qutip import *

#### layout
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True
col0, col1 = ['#f6848f', '#75a1d7']

#### define abbreviation for qutip base vectors and matrices
base = [basis(9, i) for i in range(9)]
[e0, e1, e2, e3, e4, e5, e6, e7, e8] = base

Base = [base[i]*base[i].trans() for i in range(9)]
[E0, E1, E2, E3, E4, E5, E6, E7, E8] = Base

base_names = ['$|00\\rangle$', '$|01\\rangle$', '$|0r\\rangle$', '$|10\\rangle$','$|11\\rangle$',
         '$|1r\\rangle$', '$|r0\\rangle$', '$|r1\\rangle$', '$|rr\\rangle$']

# ----------------------- Construct the pulses --------------------------

def Delta(t, args):
    """
    Piecewise constructs the detuning from the Saffman protocol.
    
    Parameters
    ----------
    t: float or ndarray
        time
    args: dict
        Dictionary of parameters:
            T (float): Total protocol duration (µs).
            Dmax (float): Maximum detuning (MHz).
            T2 (float): Waiting time (µs).
        
    Returns
    ---------
    _Delta: float or ndarray
        Detuning at time t.
    """
    T, Dmax, T2 = args["T"], args["Dmax"], args["T2"]
    
    def Delta1(t, args1):
        return - Dmax * np.cos(2 * np.pi * t / T)
    def DeltaA(t, argsA):
        return Dmax * np.cos(np.pi/T2 * (t - T/2))
    def Delta2(t, args2):
        return + Dmax * np.cos(2 * np.pi * (t-T2) / T)
    
    # Piecewise construct function
    _Delta = np.piecewise(t, [t <= T/2, (T/2 <= t) & (t <= T/2 + T2), T/2 + T2 <= t], [Delta1, DeltaA, Delta2], {})
    
    return _Delta

def dt_Delta(t, args):
    """
    Piecewise constructs the time derivative of the detuning from the Saffman protocol.
    
    Parameters
    ----------
    t: float or ndarray
        time
    args: dict
        Dictionary of parameters:
            T (float): Total protocol duration (µs).
            Dmax (float): Maximum detuning (MHz).
            T2 (float): Waiting time (µs).
        
    Returns
    ---------
    _dt_Delta: float or ndarray
        Time derivative of the detuning at time t.
    """
        
    T, Dmax, T2 = args["T"], args["Dmax"], args["T2"]
    
    def dt_Delta1(t, args1):
        return + Dmax * 2 * np.pi / T * np.sin(2 * np.pi * t / T)
    def dt_DeltaA(t, argsA):
        return - Dmax * np.pi/T2 * np.sin(np.pi/T2 * (t - T/2))
    def dt_Delta2(t, args2):
        return - Dmax * 2 * np.pi / T * np.sin(2 * np.pi * (t - T2) / T)
    
    # Piecewise construct function
    _dt_Delta = np.piecewise(t, [t <= T/2, (T/2 <= t) & (t <= T/2 + T2), T/2 + T2 <= t], [dt_Delta1, dt_DeltaA, dt_Delta2], {})
    
    return _dt_Delta


def Omega(t, args):
    """
    Piecewise constructs the Rabi pulse from the Saffman protocol.
    
    Parameters
    ----------
    t: float or ndarray
        time
    args: dict
        Dictionary of parameters:
            T (float): Total protocol duration (µs).
            Omax (float): Maximum Rabi pulse (MHz).
            tau (float): width of the Gaussian-like pulse
            T2 (float): Waiting time (µs).
        
    Returns
    ---------
    _Omega: float or ndarray
        Rabi pulse at time t.
    """    
    T, Omax, tau, T2 = args["T"], args["Omax"], args["tau"], args["T2"]
    
    a = np.exp(-(T/(4*tau))**4)
    b = -8/T*(T/4)**3/tau**4*a
    
    def Omega1(t, args1):   
        return Omax*(np.exp(-((t-T/4)/tau)**4) - a - b*t*(t-T/2))
    def Omega2(t, args2):   
        return Omax*(np.exp(-((t-3*T/4-T2)/tau)**4) - a - b*(t-T/2-T2)*(t-T-T2))
    
    # Numerically normalize the function
    its  = np.linspace(0, T/2, 200)
    Norm = Omax/np.max(Omega1(its, {}))
    
    # Piecewise construct function
    _Omega = np.piecewise(t, [t <= T/2, T/2 + T2 <= t], [Omega1, Omega2], {})*Norm
    
    return _Omega

def dt_Omega(t, args):
    """
    Piecewise constructs the time derivative of the Rabi pulse from the Saffman protocol.
    
    Parameters
    ----------
    t: float or ndarray
        time
    args: dict
        Dictionary of parameters:
            T (float): Total protocol duration (µs).
            Omax (float): Maximum Rabi pulse (MHz).
            tau (float): width of the Gaussian-like pulse
            T2 (float): Waiting time (µs).
        
    Returns
    ---------
    _Omega: float or ndarray
        Time derivative of the Rabi pulse at time t.
    """    
    T, Omax, tau, T2 = args["T"], args["Omax"], args["tau"], args["T2"]

    a = np.exp(-(T/(4*tau))**4)
    b = -8/T*(T/4)**3/tau**4*a
    
    def dt_Omega1(t, args1):   
        return Omax*(-4*(t-T/4)**3/tau**4*np.exp(-((t-T/4)/tau)**4) - b*(t-T/2) - b*t)
    
    def dt_Omega2(t, args2):   
        return Omax*(-4*(t-3*T/4-T2)**3/tau**4*np.exp(-((t-3*T/4-T2)/tau)**4) - b*(t-T-T2) - b*(t-T/2-T2))
    
    def Omega1(t, args1):   
        return Omax*(np.exp(-((t-T/4)/tau)**4) - a - b*t*(t-T/2))
    
    # Numerically normalize the function
    its = np.linspace(0, T/2, 200)
    Norm = Omax/np.max(Omega1(its, args))
    
    # Piecewise construct function
    _dt_Omega = np.piecewise(t, [t <= T/2, T/2 + T2 <= t], [dt_Omega1, dt_Omega2], {})*Norm
    
    return _dt_Omega


# ----------------------- Construct Hamiltonians --------------------------

# Auxilliary matrices
M1 = Qobj([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
M2 = Qobj([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
I3 = qeye(3)
M3 = Qobj((tensor(M1, I3) + tensor(I3, M1)).full())
M4 = Qobj((tensor(M2, I3) + tensor(I3, M2)).full())
M5 = Qobj((tensor(M2, M2)).full())

def fH_0(V, args):
    """
    Constructs the full 9x9 H_0 from Saffman et al. as a QobjEvo.
    
    Parameters
    ----------
    V: float
        time
    args: dict
        Dictionary of parameters:
            T (float): Total protocol duration (µs).
            Omax (float): Maximum Rabi pulse (MHz).
            Dmax (float): Maximum detuning (MHz).
            tau (float): width of the Gaussian-like pulse
            T2 (float): Waiting time (µs).
        
    Returns
    ---------
    _H_0: qutip.QobjEvo
        Ground Hamiltonian used in the Saffman et al. paper.
    """   
    
    _H_0 = QobjEvo([[1/2 * M3, Omega], [M4, Delta], V * M5], args)
    
    return _H_0

def fdt_H_0(V, args):
    """
    Constructs the time derivative of H_0 from Saffman et al., this is needed for H_CD.
    
    Parameters
    ----------
    V: float
        time
    args: dict
        Dictionary of parameters:
            T (float): Total protocol duration (µs).
            Omax (float): Maximum Rabi pulse (MHz).
            Dmax (float): Maximum detuning (MHz).
            tau (float): width of the Gaussian-like pulse
            T2 (float): Waiting time (µs).
        
    Returns
    ---------
    _dt_H_0: qutip.QobjEvo
        Time derivative of the ground Hamiltonian used in the Saffman et al. paper.
    """   
    
    _dt_H_0 = QobjEvo([[1/2 * M3, dt_Omega], [M4, dt_Delta]], args)
    
    return _dt_H_0


def fH_CD(H_0, dt_H_0, steps, args):
    """
    Numerically constructs a 9x9 counterdiabatic Hamiltonian for given H and its time derivative. Returns a QobjEvo that is well-defined at every time by quadratically interpolating between numerical iteration points.
    
    Parameters
    ----------
    H_0: qutip.QobjEvo
        Hamiltonian for which the counterdiabatic part is to be constructed.
    dt_H_0: qutip.QobjEvo
        Time derivative of the Hamiltonian for which the counterdiabatic part is to be constructed.
    steps: int
        Number of iteration steps at which the counterdiabatic Hamiltonian should be calculated. Between these points the final result is quadratically interpolated.
    args: dict
        Dictionary of parameters:
            T (float): Total protocol duration (µs).
        
    Returns
    ---------
    H_CD_Evo: qutip.QobjEvo
        Counterdiabatic Hamiltonian.
    """
    
    # Time array and dimension
    Ttot  = args['T'] * 1.1
    tvals = np.linspace(0, Ttot, steps)
    N     = len(base)
    
    # Empty arrays of eigenvalues and -states
    EV = np.zeros([steps, N], dtype = np.complex128)
    ES = np.zeros([N, steps, N, 1], dtype = np.complex128)
    
    # Define arrays with time evolved Hamiltonians
    H_0_vals    = [H_0(t).full() for t in tvals]
    dt_H_0_vals = [dt_H_0(t).full() for t in tvals]
    
    # Calculate eigenvalues and -states
    Eigenstates = np.linalg.eigh(H_0_vals)
    EV = np.transpose(Eigenstates[0])
    ES = np.transpose([Eigenstates[1]], axes = [3, 1, 2, 0])                                             
    ES_bra = np.conj(np.transpose(ES, axes = [0, 1, 3, 2]))
    
    # Calculate the counterdiabatic Hamiltonian
    H_CD = 0
    for n in range(N):
        for m in range(N):
            if n != m:
                proj_n = ES[n] @ ES_bra[n]
                proj_m = ES[m] @ ES_bra[m]       
                i_0 = np.where(np.abs((EV[m] - EV[n])) > 1e-10) #Numerical threshold
                H_CD_nm = np.zeros([steps, N, N], dtype = np.complex128)
                H_CD_nm[i_0] = (proj_n @ dt_H_0_vals @ proj_m)[i_0]/((EV[m] - EV[n])[i_0, None, None])
                H_CD = H_CD + 1j*H_CD_nm    
            else:
                continue
    H_CD = np.transpose(H_CD, axes = [1, 2, 0])
    H_CD_Evo = 0

    # Quadratically interpolate each matrix entry to get a QobjEvo as output
    for i in range(N):
        for j in range(N):
            Sij = Cubic_Spline(tvals[0], tvals[-1], H_CD[i][j])
            Aij = base[i]*base[j].dag()
            H_CD_Evo = H_CD_Evo + QobjEvo([Aij, Sij])
    
    return H_CD_Evo

#Numerical calculation of functions f1 and f2 
def f1_f2_num(H_0, dt_H_0, steps, args):
    """
    Numerically constructs the functions f1 nd f2 (named f_0 and f_1 in the paper). This is the same function as fH_CD but only interpolates for two matrix elements and is thus slightly faster.
    
    Parameters
    ----------
    H_0: QobjEvo
        Hamiltonian for which the counterdiabatic part is to be constructed.
    dt_H_0: QobjEvo
        Time derivative of the Hamiltonian for which the counterdiabatic part is to be constructed.
    steps: int
        Number of iteration steps at which the counterdiabatic Hamiltonian should be calculated. Between these points the final result is quadratically interpolated.
    args: dict
        Dictionary of parameters:
            T (float): Total protocol duration (µs).
        
    Returns
    ---------
    f1: qutip.interpolate.Cubic_Spline
        Function for the [1, 2] entry of H_CD, this is the function labeled as f_0 in the paper, needed to construct the effective counterdiabatic Hamiltonian.
    f2: qutip.interpolate.Cubic_Spline
        Function for the [4, 5] entry of H_CD, this is the function labeled as f_1 in the paper, needed to construct the effective counterdiabatic Hamiltonian.
    """
    
    # Time array and dimension
    Ttot  = args['T'] * 1.1
    tvals = np.linspace(0, Ttot, steps)
    N = len(base)
    
    # Empty arrays of eigenvalues and -states
    EV = np.zeros([steps, N], dtype = np.complex128)
    ES = np.zeros([N, steps, N, 1], dtype = np.complex128)
    
    # Define arrays with time evolved Hamiltonians
    H_0_vals = [H_0(t).full() for t in tvals]
    dt_H_0_vals = [dt_H_0(t).full() for t in tvals]
    
    # Calculate eigenvalues and -states
    Eigenstates = np.linalg.eigh(H_0_vals)
    EV = np.transpose(Eigenstates[0])
    ES = np.transpose([Eigenstates[1]], axes = [3, 1, 2, 0])                                             
    ES_bra = np.conj(np.transpose(ES, axes = [0, 1, 3, 2]))
    
    # Calculate the counterdiabatic Hamiltonian
    H_CD = 0
    for n in range(N):
        for m in range(N):
            if n != m:
                proj_n = ES[n] @ ES_bra[n]
                proj_m = ES[m] @ ES_bra[m]       
                i_0 = np.where(np.abs((EV[m] - EV[n])) > 1e-3)
                H_CD_nm = np.zeros([steps, N, N], dtype = np.complex128)
                H_CD_nm[i_0] = (proj_n @ dt_H_0_vals @ proj_m)[i_0]/((EV[m] - EV[n])[i_0, None, None])
                H_CD = H_CD + 1j*H_CD_nm    
            else:
                continue
    H_CD = np.transpose(H_CD, axes = [1, 2, 0])
    H_CD_Evo = 0

    # Quadratically the relevant elements for f1 and f2
    f1 = Cubic_Spline(tvals[0], tvals[-1], np.abs(H_CD[1][2]))
    f2 = Cubic_Spline(tvals[0], tvals[-1], np.abs(H_CD[4][5]))
    
    return f1, f2

# ----------------------- Construct effective counterdiabatic Hamiltonians --------------------------

# Auxilliary matrices
eta = Qobj([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
[e0_3, e1_3, e2_3] = [basis(3, i) for i in range(3)]
xi1 = e0_3 * e0_3.trans()
xi2 = e1_3 * e1_3.trans()
xi3 = e2_3 * e2_3.trans()

K1 = Qobj((tensor(eta, xi1) + tensor(xi1, eta)).full())
K2 = Qobj((tensor(eta, xi2) + tensor(xi2, eta)).full())
K3 = Qobj((tensor(xi3, I3) + tensor(I3, xi3)).full())

#numerical calculation of the coefficients
def fc1_id(f1, f2, args):
    omg = args['omg']
    def out(t, args):
        A = np.sqrt(2*np.abs(f1(t)))
        return A*np.sin(omg*t)
    return out

def fc2_id(f1, f2, args):
    omg = args['omg']
    def out(t, args):
        B = np.sqrt(2*np.abs(f2(t)))
        return B*np.cos(omg*t)
    return out

def fc3_id(f1, f2, args):
    omg = args['omg']
    def out(t, args):
        A = np.sqrt(2*np.abs(f1(t)))
        B = np.sqrt(2*np.abs(f2(t)))
        return A*np.cos(omg*t)-B*np.sin(omg*t)
    return out

def fH_E_id(f1, f2, args):
    """
    Constructs the full 9x9 effective counterdiabatic Hamiltonian as a QobjEvo.
    
    Parameters
    ----------
    f1: qutip.interpolate.Cubic_Spline
        Function for the [1, 2] entry of H_CD, this is the function labeled as f_0 in the paper, needed to construct the effective counterdiabatic Hamiltonian.
    f2: qutip.interpolate.Cubic_Spline
        Function for the [4, 5] entry of H_CD, this is the function labeled as f_1 in the paper, needed to construct the effective counterdiabatic Hamiltonian.
    args: dict
        Dictionary of parameters:
            V (float): Rydberg blockade (MHz).
            omg (float): eCD frequency (MHz).
        
    Returns
    ---------
    _H_E_id: qutip.QobjEvo
        Effective counterdiabatic Hamiltonian.
    """

    V, omg = args['V'], args['omg']

    # Get coefficients
    c1_id = fc1_id(f1, f2, args)
    c2_id = fc2_id(f1, f2, args)
    c3_id = fc3_id(f1, f2, args)
    
    # Effective counterdiabatic Hamiltonian
    _H_E_id = np.sqrt(omg) * QobjEvo([[K1, c1_id], [K2, c2_id], [K3, c3_id], V * M5], args = args)

    return _H_E_id

def propH_E_id(f1, f2, steps, args):
    """
    Constructs the propagator of the effective counterdiabatic Hamiltonian.
    
    Parameters
    ----------
    f1: qutip.interpolate.Cubic_Spline
        Function for the [1, 2] entry of H_CD, this is the function labeled as f_0 in the paper, needed to construct the effective counterdiabatic Hamiltonian.
    f2: qutip.interpolate.Cubic_Spline
        Function for the [4, 5] entry of H_CD, this is the function labeled as f_1 in the paper, needed to construct the effective counterdiabatic Hamiltonian.
    steps: int
        Number of iteration steps at which the counterdiabatic Hamiltonian should be calculated.
    args: dict
        Dictionary of parameters:
            V (float): Rydberg blockade (MHz).
            omg (float): eCD frequency (MHz).
            T (float): Total protocol duration (µs).
        
    Returns
    ---------
    _propH_E_id: numpy.ndarray
        Array containing the propagator at every time defined by steps.
    """
    V, omg = args['V'], args['omg']
    Ttot   = args['T'] * 1.1

    # Get coefficients
    c1_id = fc1_id(f1, f2, args)
    c2_id = fc2_id(f1, f2, args)
    c3_id = fc3_id(f1, f2, args)
    
    # List of components of the effective counterdiabatic Hamiltonian
    listH_E_id  = [[np.sqrt(omg) * K1, c1_id], [np.sqrt(omg) * K2, c2_id], [np.sqrt(omg) * K3, c3_id], V * M5]
    tvals       = np.linspace(0, Ttot, steps)
    _propH_E_id = propagator(listH_E_id, tvals, args=args)

    return _propH_E_id

def propH_E_id_err(f1, f2, c12_err, d_err, steps, args):
    """
    Constructs the propagator of the effective counterdiabatic Hamiltonian with errors on the Rabi amplitude and detuning.
    
    Parameters
    ----------
    f1: qutip.interpolate.Cubic_Spline
        Function for the [1, 2] entry of H_CD, this is the function labeled as f_0 in the paper, needed to construct the effective counterdiabatic Hamiltonian.
    f2: qutip.interpolate.Cubic_Spline
        Function for the [4, 5] entry of H_CD, this is the function labeled as f_1 in the paper, needed to construct the effective counterdiabatic Hamiltonian.
    c12_err: float
        Relative error on the Rabi pulse.
    d_err: float
        Absolute error on the detuning.
    steps: int
        Number of iteration steps at which the counterdiabatic Hamiltonian should be calculated.
    args: dict
        Dictionary of parameters:
            V (float): Rydberg blockade (MHz).
            omg (float): eCD frequency (MHz).
            T (float): Total protocol duration (µs).
        
    Returns
    ---------
    _propH_E_id: numpy.ndarray
        Array containing the propagator at every time defined by steps.
    """
    V, omg = args['V'], args['omg']
    Ttot   = args['T'] * 1.1

    # Get coefficients
    c1_id = fc1_id(f1, f2, args)
    c2_id = fc2_id(f1, f2, args)
    c3_id = fc3_id(f1, f2, args)
    
    # List of components of the effective counterdiabatic Hamiltonian
    listH_E_id  = [[np.sqrt(omg) * (1 + c12_err) * K1, c1_id], 
                   [np.sqrt(omg) * (1 + c12_err) * K2, c2_id], 
                   [np.sqrt(omg) * K3, c3_id], d_err * K3 + V * M5]
    tvals       = np.linspace(0, Ttot, steps)
    _propH_E_id = propagator(listH_E_id, tvals, args=args)

    return _propH_E_id

# ----------------------- Define perfect CZ gate --------------------------

def propagator_fid(U, U_target):
    """
    Calculates the gate fidelity for griven input and target qubit system operation matrix.
    
    Parameters
    ----------
    U: ndarray
        Propagation matrix representing the effective action of the achieved gate on a qubit system.
    U_target: ndarray
        Targeted operation that was supposed to be realized with U.
        
    Returns
    ---------
    _propagator_fid: float
        Achieved gate fidelity.
    """
    
    _propagator_fid = np.abs((U_target.dag() * U.dag()).tr() / 
                  (U_target.dag() * U_target.dag()).tr()
                 ) ** 2
    
    return _propagator_fid

# Construct the ideal CZ gate
CZ_perfect = np.zeros((9,9))
CZ_perfect[0, 0] = 1
CZ_perfect[1, 1] = -1
CZ_perfect[3, 3] = -1
CZ_perfect[4, 4] = -1
CZ_perfect = Qobj(CZ_perfect)