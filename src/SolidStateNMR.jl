# Spinning sideband intensity calculations based on
# J. Herzfeld and A. E. Berger, J. Chem. Phys. 1980, 73, 6021–6030.

module SolidStateNMR

using Cubature

# τ functions
τ₋(α,β,θ) = 1/24*cos(2α)*(3 + cos(2β))*sin(2θ) -
            1/6*sin(2α)*cos(β)*cos(2θ) +
            √2/6*cos(2α)sin(2β)sin(θ) -
            √2/3*sin(2α)*sin(β)*cos(θ)
τ₊(β,θ) = 1/24*(cos(2β)-1)*sin(2θ) +
          √2/6*sin(2β)*sin(θ)

# Δ functions, σ values in ppm
Δ₋(Ω,ωᵣ,σxx,σyy) = -Ω/ωᵣ*(σxx - σyy)*1.0e-6
Δ₊(Ω,ωᵣ,σiso,σzz) = -3Ω/ωᵣ*(σiso - σzz)*1.0e-6

# Integrand to F function
f(θ,N,α,β,Δ₊,Δ₋) = 1/(2π)*exp(im*(-N*θ + Δ₋*τ₋(α,β,θ) + Δ₊*τ₊(β,θ)))

function F(N,α,β,Δ₊,Δ₋;maxevals=150)
    (val1,err1) = pquadrature(θ->real(f(θ,N,α,β,Δ₊,Δ₋)),0.0,2π;maxevals=maxevals)
    (val2,err2) = pquadrature(θ->imag(f(θ,N,α,β,Δ₊,Δ₋)),0.0,2π;maxevals=maxevals)
    return val1+im*val2
end

# Relative intensity of the Nth sideband
intensity(N,Δ₊,Δ₋;maxevals=1000) = hcubature(x->sin(x[2])*abs2(F(N,x[1],x[2],Δ₊,Δ₋)),[0,0],[2π,π];maxevals=maxevals)[1]/4π

# List of sideband intensities:
# -maxN, -maxN + 1, ..., -1, 0, 1, ..., maxN - 1, maxN 
# σ: eigenvalues of the shielding tensor in ppm
# ωᵣ: rotor speed in Hz
# Ω: B₀ in Hz
# maxN: number of sidebands calculated on either side of the main peak
function sidebands(Ω,ωᵣ,σ; maxN=3, maxevals=500)
    σiso = (σ[1] + σ[2] + σ[3])/3
    D₊ = Δ₊(Ω,ωᵣ,σiso,σ[3])
    D₋ = Δ₋(Ω,ωᵣ,σ[1],σ[2])
    Is = [intensity(N, D₊, D₋;maxevals=maxevals) for N=0:maxN]
    [reverse(Is[2:end]);Is]  # I[-n] = I[n]
end

# Lorentzian function
# a: peak maximum
# Γ: full width at half height
# x₀: center
# Maximum: 2a/(π*Γ)
L(a,x₀,Γ) = x -> Γ*a/(2π*((x-x₀)^2+(Γ/2)^2))

# Composition of Lorentzian functions
# a: amplitudes
# f₀: center frequencies
# w: peak widths at half height
function compose(a,f₀,w)
    funcs = [L(x[1],x[2],x[3]) for x=zip(a,f₀,w)]
    f -> sum(l(f) for l=funcs)
end

# Solid state NMR spectrum for a single atom
# σ: Eigenvalues of the shielding tensor in ppm
# ref: chemical shift reference (used for spectrum alignment)
# Ω: B₀ in Hz
# ωᵣ: rotor speed in Hz
# w: peak width at half height
# maxN: number of sidebands calculated on either side of the main peak
function spectrum{T<:Number}(σ::Array{T,1},ref,Ω,ωᵣ,w; maxN=3)
    ref_freq = ref*1.0e-6*Ω # reference in Hz
    center = ref - mean(σ)*1.0e-6*Ω # main peak location in Hz
    a = sidebands(Ω,ωᵣ,σ; maxN=maxN)
    f₀ = [center+(ωᵣ*n) for n=-maxN:maxN]
    Γ = w * ones(2*maxN+1)
    compose(a,f₀,Γ)
end

# Solid state NMR spectrum for a collection of atoms
# (This is the function you most likely want to use)
# atoms: [("atom_label1", σ₁), ("atom_label1", σ₂) , ...]
# where σᵢ is the 3-element array of the shielding tensor
# eigenvalues for atom i
# ref: chemical shift reference (used for spectrum alignment)
# Ω: B₀ in Hz
# ωᵣ: rotor speed in Hz
# w: peak width at half height
# maxN: number of sidebands calculated on either side of the main peak
function spectrum{T<:Number,S<:AbstractString}(atoms::Array{Tuple{S,Array{T,1}},1},ref_ppm,Ω,ωᵣ,w; maxN=3)
    spects = pmap(atom->spectrum(atom[2],ref_ppm,Ω,ωᵣ,w; maxN=maxN),atoms)
    x->sum(s(x) for s=spects)
end
end