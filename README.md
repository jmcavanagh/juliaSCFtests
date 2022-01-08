# juliaSCFtests
A few electronic structure tools written in Julia. This is a Julia implementation of minimal basis set Hartree Fock (as written in Szabo and Ostlund) and Full Configuration Interaction of the H2 Dimer written in Julia. 
HartreeFock.jl contains all of the machinery for Hartree Fock, including the integrals.  
fci.jl calls HartreeFock.jl to perform HF calculations on the H<sub>2</sub> dimer at a variety of distances, and plots these compared with the energ found by FCI at this length.  
MolecularOrbitalHeatmap.jl visualizes the cross section of the bonding and antibonding orbitals of H<sub>2</sub>.  

To run on the julia REPL, run ```include("fci.jl")```, for example
