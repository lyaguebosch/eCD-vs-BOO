# eCD vs BOO comparison

Code for the paper:  
[Yet to be published]

---

## Overview

This repository contains the source code used in our simulations for the paper.  
Specifically, it includes:

- Functions to recreate the original Saffman protocol
- Functions to construct counterdiabatic and effective counterdiabatic (eCD) Hamiltonians
- Functions for numerical optimization using the Boulder Opal (BOO) Package
- Simulations of the eCD and BOO gate for different protocol times and pulse scalings
- Stability analysis of the eCD and BOO gates
- Notebooks to visually compare both gates and recreate the plots used in the paper

---

## Requirements

This project uses Python ≥ 3.8. 
This project was done with QuTip version 4.7.6 and is incompatible with 5.0 or higher 
Dependencies are listed in requirements.txt

---

## References

The code is based on a gate originally proposed by Saffman et al. [1]. The gate was modified using the code partially available here using shortcut-toadiabaticity techniques [2]. This approach is compared to a numerical optimization using the Q-ctrl Boulder Opal package, discussed in detail in [3]. Parts of the code used here are directly taken from [3].

[1] M. Saffman, I. I. Beterov, A. Dalal, E. J. Páez, and B. C. Sanders, Symmetric Rydberg controlled-Z gates with adiabatic pulses, Phys. Rev. A 101, 062309 (2020).

[2] L. S. Yagüe Bosch, T. Ehret, F. Petiziol, E. Arimondo, and S. Wimberger, Shortcut-to-Adiabatic Controlled- Phase Gate in Rydberg Atoms, Annalen der Physik 535, 2300275 (2023).

[3] Q-CTRL, Boulder Opal documentation: Design ro- bust rydberg blockade two-qubit gates in cold atoms, https://docs.q-ctrl.com/boulder-opal/apply/rydberg-atom-quantum-computing/design-robust-rydberg- blockade-two-qubit-gates-in-cold-atoms (2022),
