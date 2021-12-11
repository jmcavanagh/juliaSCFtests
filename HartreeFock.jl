module HartreeFock
using LinearAlgebra
using SpecialFunctions


struct prim_gauss
    ex::Float64
    R::Vector{Float64}
end
export prim_gauss

struct basis_function
    coeffs::Vector{Float64}
    gauss::Vector{prim_gauss}
end
export basis_function

import Base.length

length(bf::basis_function) = length(bf.coeffs)

function overlap_prim_integral(pg1::prim_gauss, pg2::prim_gauss)
    α, β = pg1.ex, pg2.ex
    RA, RB = pg1.R, pg2.R
    return (pi / (α + β))^1.5 * exp(-norm2(RA - RB) * α * β / (α + β))
end

function F_0(t)
    if t == 0
        return 1.0
    else
        return 0.5 * √(π / t) * erf(√t)
    end
end

norm2(V) = sum(abs2, V)

function e1_prim_integral(pg1::prim_gauss, pg2::prim_gauss, RC, ZC)
    α, β = pg1.ex, pg2.ex
    RA, RB = pg1.R, pg2.R
    H =
        α * β / (α + β) *
        (3 - 2 * α * β / (α + β) * norm2(RA - RB)) *
        overlap_prim_integral(pg1, pg2) / 2
    RP = (α * RA + β * RB) / (α + β)
    t = (α + β) * norm2(RP - RC)
    H += -2 * π / (α + β) * ZC * exp(-α * β / (α + β) * norm2(RA - RB)) * F_0(t)

end

function e2_prim_integral(
    pg1::prim_gauss,
    pg2::prim_gauss,
    pg3::prim_gauss,
    pg4::prim_gauss,
)
    α, β, γ, δ = pg1.ex, pg2.ex, pg3.ex, pg4.ex
    RA, RB, RC, RD = pg1.R, pg2.R, pg3.R, pg4.R
    E2 = 2 * π^2.5 / ((α + β) * (γ + δ) * (α + β + γ + δ)^0.5)
    E2 *= exp(-α * β / (α + β) * norm2(RA - RB) - γ * δ / (γ + δ) * norm2(RC - RD))
    t =
        (α + β) * (γ + δ) / (α + β + γ + δ) *
        norm2((α * RA + β * RB) / (α + β) - (γ * RC + δ * RD) / (γ + δ))
    E2 *= F_0(t)
    return E2
end

function overlap_integral(bf1::basis_function, bf2::basis_function)
    o = 0
    for i = 1:length(bf1), j = 1:length(bf2)
        o +=
            bf1.coeffs[i] *
            bf2.coeffs[j] *
            overlap_prim_integral(bf1.gauss[i], bf2.gauss[j])
    end
    return o
end

function e1_integral(bf1::basis_function, bf2::basis_function, RC, ZC = 1)
    e1 = 0
    for i = 1:length(bf1), j = 1:length(bf2)
        e1 +=
            bf1.coeffs[i] *
            bf2.coeffs[j] *
            e1_prim_integral(bf1.gauss[i], bf2.gauss[j], RC, ZC)
    end
    return e1
end

function e2_integral(
    bf1::basis_function,
    bf2::basis_function,
    bf3::basis_function,
    bf4::basis_function,
)
    e2 = 0
    for i = 1:length(bf1), j = 1:length(bf2), k = 1:length(bf3), l = 1:length(bf4)
        e2 +=
            bf1.coeffs[i] *
            bf2.coeffs[j] *
            bf3.coeffs[k] *
            bf4.coeffs[l] *
            e2_prim_integral(bf1.gauss[i], bf2.gauss[j], bf3.gauss[k], bf4.gauss[l])
    end
    return e2
end

function overlap_matrix(phi_vector)
    S = zeros(2, 2)
    for j = 1:2, k = 1:2
        S[j, k] = overlap_integral(phi_vector[j], phi_vector[k])
    end
    return S
end

function e1_matrix(phi_vector)
    H = zeros(2, 2, 2)
    for j = 1:2, k = 1:2, l = 1:2
        H[j, k, l] = e1_integral(phi_vector[j], phi_vector[k], phi_vector[l].gauss[1].R)
    end
    return sum(H, dims = 3)[:,:,1]
end

function e2_array(phi_vector::Vector{basis_function})
    E2 = zeros(2, 2, 2, 2)
    for i = 1:2, j = 1:2, k = 1:2, l = 1:2
        E2[i, j, k, l] =
            e2_integral(phi_vector[i], phi_vector[j], phi_vector[k], phi_vector[l])
    end
    return E2
end

struct hf_output
    S::Array{Float64, 2}
    X::Array{Float64, 2}
    H::Array{Float64, 2}
    E2::Array{Float64, 4}
    C::Array{Float64, 2}
    ϵ::Array{Float64, 1}
    energy::Float64
end

function hartree_fock(phi_vector, convergence_criterion, C = ones(2,2))
    S = overlap_matrix(phi_vector)
    X = S^(-1 / 2)
    H = e1_matrix(phi_vector)
    E2 = e2_array(phi_vector)
    ϵ = zeros(2)

    P_old = ones(2,2)

    for n = 1:10
        P = zeros(2, 2)
        for i = 1:2, j = 1:2
            P[i, j] = 2C[i, 1] * C[j, 1]
        end
        if abs(sum(P_old) - sum(P)) < convergence_criterion
           break
        end
        F = zeros(2, 2)
        for μ = 1:2, ν = 1:2
            F[μ, ν] += H[μ, ν]
            for λ = 1:2, σ = 1:2
                F[μ, ν] += P[λ, σ] * (E2[μ, ν, σ, λ] - 0.5 * E2[μ, λ, σ, ν])
            end
        end
        ϵ, CC = eigen(X' * F * X)
        C = X * CC
        P_old = P
    end
    h_11 = 0
    for μ in 1:2, ν in 1:2
        h_11 += H[μ, ν] * C[μ, 1] * C[ν, 1]
    end
    J_11 = 0
    for μ in 1:2, ν in 1:2, λ in 1:2, δ in 1:2
        J_11 += C[μ,1] * C[ν,1] * C[λ,1] * C[δ,1] * E2[μ, ν, λ, δ]
    end
    energy = 2*h_11 + J_11
    for μ in 1:length(phi_vector), ν in 1:μ
        if μ != ν
            energy += 1/(sum(abs,phi_vector[μ].gauss[1].R - phi_vector[ν].gauss[1].R))
        end
    end
    return hf_output(S, X, H, E2, C, ϵ, energy)
    #return hf_output(S, X, H, E2, C, ϵ, energy)
end
export hartree_fock

end
