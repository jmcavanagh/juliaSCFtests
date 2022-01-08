push!(LOAD_PATH, ".")
using HartreeFock
using LinearAlgebra
using Plots

struct bond_energies
    distances::Array{Float64,1}
    hf_energies::Array{Float64,1}
    fci_energies::Array{Float64,1}
end

function find_bond_potential(min_length, step_size, max_length)

    STO_1G_EXP = [0.270950]
    STO_1G = [(0.270950 * 2 / π)^0.75]
    ZC = 1

    STO_2G = [0.678914, 0.430129]
    STO_2G_EXP = [0.151623, 0.851819]

    STO_3G_EXP = [0.168856, 0.623913, 3.42525]
    STO_3G = [0.444635, 0.535328, 0.154329] .* (2 * STO_3G_EXP / π) .^ 0.75

    number_of_distances = Int(floor((max_length - min_length) / step_size))
    total_energies = zeros(number_of_distances)
    fci_energies = zeros(number_of_distances)
    distances = zeros(number_of_distances)

    convergence_criterion = 0.0001


    for m = 1:number_of_distances
        dist = round(min_length + step_size * m, digits = 3)
        R_A = [0, 0, 0]
        R_B = [0, 0, dist]
        global phi_vector = [
            basis_function(
                STO_3G,
                [
                    prim_gauss(STO_3G_EXP[1], R_A),
                    prim_gauss(STO_3G_EXP[2], R_A),
                    prim_gauss(STO_3G_EXP[3], R_A),
                ],
            ),
            basis_function(
                STO_3G,
                [
                    prim_gauss(STO_3G_EXP[1], R_B),
                    prim_gauss(STO_3G_EXP[2], R_B),
                    prim_gauss(STO_3G_EXP[3], R_B),
                ],
            ),
        ]
        output = hartree_fock(phi_vector, convergence_criterion)
        distances[m] = dist
        total_energies[m] = output.energy

        #Full CI for H2: getting the best BDE for the STO-3G basis set!
        C = output.C
        h_11 = 0
        for μ = 1:2, ν = 1:2
            h_11 += output.H[μ, ν] * C[μ, 1] * C[ν, 1]
        end
        h_22 = 0
        for μ = 1:2, ν = 1:2
            h_22 += output.H[μ, ν] * C[μ, 2] * C[ν, 2]
        end
        J_11 = 0
        for μ = 1:2, ν = 1:2, λ = 1:2, δ = 1:2
            J_11 += C[μ, 1] * C[ν, 1] * C[λ, 1] * C[δ, 1] * output.E2[μ, ν, λ, δ]
        end
        J_12 = 0
        for μ = 1:2, ν = 1:2, λ = 1:2, δ = 1:2
            J_12 += C[μ, 1] * C[ν, 1] * C[λ, 2] * C[δ, 2] * output.E2[μ, ν, λ, δ]
        end
        J_22 = 0
        for μ = 1:2, ν = 1:2, λ = 1:2, δ = 1:2
            J_22 += C[μ, 2] * C[ν, 2] * C[λ, 2] * C[δ, 2] * output.E2[μ, ν, λ, δ]
        end
        K_12 = 0
        for μ = 1:2, ν = 1:2, λ = 1:2, δ = 1:2
            K_12 += C[μ, 1] * C[ν, 2] * C[λ, 1] * C[δ, 2] * output.E2[μ, ν, λ, δ]
        end
        FCI = [2*h_11+J_11 K_12; K_12 2*h_22+J_22]
        z = eigvals(FCI)
        #println(z)
        fci_energies[m] = z[1] + 1/dist
    end
    return bond_energies(distances, total_energies, fci_energies)
end
min_length = 0.4
step_size = 0.01
max_length = 6.0

bond = find_bond_potential(min_length, step_size, max_length)
plot(bond.distances, bond.hf_energies, label="Hartree-Fock")
plot!(bond.distances, bond.fci_energies, label="Full CI")
xlabel!("Distance (AU)")
ylabel!("Energy (Hartree)")


