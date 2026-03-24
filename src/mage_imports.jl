using Dates

using Serialization
using Base.Threads
using LinearAlgebra
import JSON
@show pwd()
nt = nthreads()
@show nt
@show BLAS.get_num_threads()
BLAS.set_num_threads(1)
@show BLAS.get_num_threads()
@show Sys.CPU_THREADS

using Dates
import SearchNetworks as sn
import DataStructures: OrderedDict
using UUIDs
using Images, FileIO
using ImageCore
using ImageBinarization
using CSV
using DataFrames
using Statistics
using Random
using StatsBase: sample
using Flux
using Logging
using HDF5
using DataFlowTasks
using ArgParse
import PNGFiles
using ThreadPinning
# using MAGENetwork
using PythonCall
using TiffImages
using PythonCall
using DelimitedFiles
using DistributedNext
import StatsBase: countmap
using JLD2
using UnicodePlots

function fast_asinh(x)
    abs_x = abs(x)
    sign_x = sign(x)

    if abs_x < 0.5
        # Near-zero approximation: asinh(x) ≈ x
        return x
    else
        # Approximation for larger values: sign(x) * log(1 + |x|)
        return sign_x * log(1 + abs_x)
    end
end
fast_asinh(x::AbstractArray) = broadcast(y -> fast_asinh(y), x)
Flux.tanh(x::AbstractArray) = broadcast(y -> Flux.tanh(y), x)
