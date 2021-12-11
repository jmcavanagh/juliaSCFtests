push!(LOAD_PATH, ".")

using HartreeFock
using Plots


STO_3G_EXP = [0.168856, 0.623913, 3.42525]
STO_3G = [0.444635, 0.535328, 0.154329] .* (2 * STO_3G_EXP / π) .^ 0.75


dist = 1.2

R_A = [0, 0, 0]
R_B = [dist, 0, 0]


x = LinRange(-1.8*dist, 2.8 * dist, 400)
y = LinRange(-1.8 * dist, 1.8 * dist, 400)
z = 0

phi_vector = [
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
hf = hartree_fock(phi_vector, 0.0001)
#hf = hartree_fock(phi_vector, 0.0001, [1.0 0.0; 0.0 1.0])
C = hf.C

function OMO(x, y, z)
    omo = 0
    for μ = 1:2
        for k = 1:3
            omo +=
                phi_vector[μ].coeffs[k] * C[μ, 1] *
                exp(-sum(abs2, phi_vector[μ].gauss[k].R - [x, y, z]) * STO_3G_EXP[k])
        end
    end
    return omo
end

function UMO(x, y, z)
    omo = 0
    for μ = 1:2
        for k = 1:3
            omo +=
                phi_vector[μ].coeffs[k] * C[μ, 2] *
                exp(-sum(abs2, phi_vector[μ].gauss[k].R - [x, y, z]) * STO_3G_EXP[k])
        end
    end
    return omo
end

omo = zeros(length(x), length(y))
umo = zeros(length(x), length(y))
for a = 1:length(x)
    for b = 1:length(y)
        omo[a, b] = round(OMO(x[a], y[b], z), digits = 5)
        umo[a, b] = round(UMO(x[a], y[b], z), digits = 5)
    end
end
println(hf.energy)
println(hf.energy)
println(hf.energy)

heatmap(hcat(omo, umo), color=cgrad([:red3, :black, :blue]))
