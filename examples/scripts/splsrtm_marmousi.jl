# Sparisty-promoting LS-RTM of the 2D Marmousi model with on-the-fly Fourier transforms
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using Statistics, Random, LinearAlgebra
using JUDI, SegyIO, HDF5, JOLI, PyPlot, SlimOptim

# Load migration velocity model
if ~isfile("marmousi_model.h5")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_model.h5`)
end
n, d, o, m0 = read(h5open("marmousi_model.h5", "r"), "n", "d", "o", "m0")

# Set up model structure
model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)

# Load data
if ~isfile("marmousi_2D.segy")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_2D.segy`)
end
block = segy_read("marmousi_2D.segy")
d_lin = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.03)    # 30 Hz wavelet
q = judiVector(src_geometry, wavelet)

# Set up info structure
ntComp = get_computational_nt(q.geometry, d_lin.geometry, model0)  # no. of computational time steps
info = Info(prod(model0.n), d_lin.nsrc, ntComp)

###################################################################################################

# Setup operators
opt = Options(optimal_checkpointing=false)

batchsize = 10
# Right-hand preconditioners (model topmute)
Mr = judiTopmute(model0.n, 52, 10)

C = joCurvelet2D(model0.n[1], model0.n[2]; zero_finest = true, DDT = Float32, RDT = Float32)

function breg_obj(x)
    i = randperm(d_lin.nsrc)[1:batchsize]
    Ml = judiMarineTopmute2D(30, d_lin[i].geometry)
    # Ml is expected to be a JOLI linear operator (joAbstractLinearOperator) PR welcome for extensions.
    # Mr is expected to be a linear operator (JOLI) or a matrix supporting Mr*x and transpose(Mr)
    f, g = lsrtm_objective(model0, q[i], d_lin[i], x; options=opt, dprecon=Ml, mprecon=Mr)
    return f, g[1:end]
end

bregopt = bregman_options(maxIter=20, verbose=2, quantile=.9, alpha=1, antichatter=true)
sol = bregman(breg_obj, 0f0.*vec(m0), bregopt, C)