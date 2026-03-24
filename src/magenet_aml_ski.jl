using Dates
st = time()
using CUDA
# CUDA.math_mode!(CUDA.FAST_MATH)
using cuDNN
@info CUDA.versioninfo()
@info CUDA.runtime_version()

using Serialization
using Base.Threads
using LinearAlgebra
using InteractiveUtils
using BenchmarkTools
import JSON
@show pwd()
nt = nthreads()
@show nt
@show BLAS.get_num_threads()
BLAS.set_num_threads(1)
@show BLAS.get_num_threads()
@show Sys.CPU_THREADS
Sys.CPU_THREADS = nt
@show Sys.CPU_THREADS

using HypothesisTests
using Revise
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
using MLDatasets
using MLDataUtils
using MAGENetwork
using PythonCall
using TiffImages
using PythonCall
using DelimitedFiles
using Distributed
import StatsBase: countmap

ENV["JULIA_DEBUG"] = "UTCGP"
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))
if occursin("REPL", @__FILE__)
    Parsed_args = Dict(
        "seed" => 1,
        "output_dir" => "./",
        "trial_id" => string(UUIDs.uuid4()),
        "mutation_rate" => 1,
        "mutation_n_models" => 1,
        "n_nodes" => 7,
        "n_elite" => 10,
        "n_new" => 50,
        "tour_size" => 40,
        "n_samples" => 100,
        "n_repetitions" => 1,
        "budget" => 6_000,
        "err_w" => 0.9,
        "time_w" => 100,
        "generations_mage" => 20,
        "lambda_mage" => 8,
        "trainsize_mage" => 400,
        # GPU
        "device" => 0,
        #NN
        "epochs_surr" => 200,
        "n_surr" => 3,
        "n_nn_batches_train" => 8,
        "n_nn_batches_val" => 20,
        # ARCH
        "n_layers" => 3,
        "n_layers_img" => 1,
        "l1_size" => 3,
        "l2_size" => 3,
        "l3_size" => 6,
        "l4_size" => 3,

        # MAGENetwork
        "th_for_align" => .05, # when to trigger MAGE → NN alignment (trigger if above)
        "th_for_es" => 1000., # when to stop NN → MAGE alignment
        "reset_pb" => 0.05, # when to stop NN → MAGE alignment
        "dropout_rate" => 0.4, # For dense layers
        "freeze_rate" => 0.8, # For NNnet (ex : 20% without gradient updates)
        "batch_size" => 256, # batch size for everything
        "regularization" => 0.01f0, # Strength of opnorm regularization 
        "inter_losses" => 1.0f0, # Strength of opnorm regularization 
        "mask" => "false", # TODO
        "optim" => "lion",
        "act" => "tanh", # or tanh
        "max" => 1, # or tanh
        # libs
        "use_ski" => "false",
    )
    global rootdir
    rootdir = "./"
    @warn "Running in repl with settings : $Parsed_args"
else
    s = ArgParseSettings()
    @add_arg_table s begin
        "--seed"
        arg_type = Int
        default = 42
        "--output_dir"
        arg_type = String
        default = "/tmp"
        "--trial_id"
        arg_type = String
        default = ""
        "--mutation_rate"
        arg_type = Int
        default = 1
        "--mutation_n_models"
        arg_type = Int
        default = 1
        "--n_nodes"
        arg_type = Int
        default = 5
        "--n_new"
        arg_type = Int
        default = 500
        "--tour_size"
        arg_type = Int
        default = 5
        "--n_samples"
        arg_type = Int
        default = 150
        "--n_repetitions"
        arg_type = Int
        default = 5
        "--err_w"
        arg_type = Float64
        default = 0.8
        "--time_w"
        arg_type = Float64
        default = 100.0
        # MAGE FITTING
        "--generations_mage"
        arg_type = Int
        default = 10
        "--lambda_mage"
        arg_type = Int
        default = 5
        "--trainsize_mage"
        arg_type = Int
        default = 30
        # CUDA
        "--device"
        arg_type = Int
        default = 0
        # NN
        "--epochs_surr"
        arg_type = Int
        default = 30
        "--n_surr"
        arg_type = Int
        default = 1
        "--n_nn_batches_train"
        arg_type = Int
        default = 2
        "--n_nn_batches_val"
        arg_type = Int
        default = 2

        # architecture
        "--n_layers"
        arg_type = Int
        default = 2
        "--n_layers_img"
        arg_type = Int
        default = 1
        "--l1_size"
        arg_type = Int
        default = 2
        "--l2_size"
        arg_type = Int
        default = 2
        "--l3_size"
        arg_type = Int
        default = 2
        "--l4_size"
        arg_type = Int
        default = 2

        # ALIGN
        "--th_for_align"
        arg_type = Float64
        "--th_for_es"
        arg_type = Float64
        "--reset_pb"
        arg_type = Float64
        "--dropout_rate"
        arg_type = Float64
        "--freeze_rate"
        arg_type = Float64
        "--batch_size"
        arg_type = Int
        default = 256
        "--regularization"
        arg_type = Float32
        "--inter_losses"
        arg_type = Float32
        "--mask"
        arg_type = String
        default = "false"
        "--optim"
        arg_type = String
        "--act"
        arg_type = String
        "--max"
        arg_type = Int
       
        # LIBS
        "--use_ski"
        arg_type = String
        default = "false"
    end
    global rootdir
    rootdir = "./"
    Parsed_args = parse_args(s)
    @show Parsed_args
end


const USE_SKI = Parsed_args["use_ski"] == "true"
@info "SKIMAGE : $USE_SKI"
const SEED::Int = Parsed_args["seed"]
Random.seed!(SEED)
const ERR_WEIGHT::Float64 = Parsed_args["err_w"]
const NERR_WEIGHT::Float64 = 1.0 - ERR_WEIGHT
const TIMEPENALTY::Int = Parsed_args["time_w"]
const N_REPETITIONS = Parsed_args["n_repetitions"]
const NSAMPLES::Int = Parsed_args["n_samples"]
const GENS::Int = 10000
const HLIMIT::Float64 = 1

const N_NN_BATCHES_TRAIN::Int = Parsed_args["n_nn_batches_train"]
const N_NN_BATCHES_VAL::Int = Parsed_args["n_nn_batches_val"]
const DEVICE_CU::Int = Parsed_args["device"]
const K::Int = Parsed_args["n_surr"]
const EPOCHS_SURR::Int = Parsed_args["epochs_surr"]
const N_LAYERS::Int = Parsed_args["n_layers"]
const N_LAYERS_IMG::Int = Parsed_args["n_layers_img"]
const L1_SIZE::Int = Parsed_args["l1_size"]
const L2_SIZE::Int = Parsed_args["l2_size"]
const L3_SIZE::Int = Parsed_args["l3_size"]
const L4_SIZE::Int = Parsed_args["l4_size"]

# MAGE FITTING
const GENERATIONS_MAGE::Int = Parsed_args["generations_mage"]
const LAMBDA_MAGE::Int = Parsed_args["lambda_mage"]
const TRAINSIZE_MAGE::Int = Parsed_args["trainsize_mage"]

# GA
const MUTATION_RATE::Int = Parsed_args["mutation_rate"]
const MUTATION_N_MODELS::Int = Parsed_args["mutation_n_models"]
const N_NODES::Int = Parsed_args["n_nodes"]
const N_NEW::Int = Parsed_args["n_new"]
const TOUR_SIZE::Int = Parsed_args["tour_size"]

# MAGENetwork

const TH_FOR_ALIGN::Float64 = Parsed_args["th_for_align"] 
const TH_FOR_ES::Float64 = Parsed_args["th_for_es"] 
const RESET_PB::Float64 = Parsed_args["reset_pb"] 
const DROPOUT_RATE::Float64 =Parsed_args["dropout_rate"]
const FREEZE_RATE::Float64 =Parsed_args["freeze_rate"]
const BATCH_SIZE::Int =Parsed_args["batch_size"]
const REGULARIZATION::Float32 =Parsed_args["regularization"]
const INTER_LOSSES::Float32 =Parsed_args["inter_losses"]
const MASK::Bool = Parsed_args["mask"] == "true" # TODO
const OPTIM::String= Parsed_args["optim"]
const ACT::Bool= Parsed_args["act"] == "tanh"
const MAX::Int= Parsed_args["max"]
        
const OUTPUTDIR::String = Parsed_args["output_dir"]
const ID::String = Parsed_args["trial_id"]

println(threadinfo(; slurm=true, color=false))
if USE_SKI
    addprocs(nt, exeflags=["--threads=1"])
end
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images
# pinthreads(:cores)
println(threadinfo(; slurm=true, color=false))
const dataset_base_path::String = "dataset_aml"
const annot_base_path::String = "AML-CYTOMORPHOLOGY_LMU-annotations-dat"
const W::Int = 32
const H::Int = W
const NCLASSES = 2
rootdir = pwd()

MAGENetwork.TH[] = TH_FOR_ALIGN 
MAGENetwork.TH_ES[] = TH_FOR_ES
MAGENetwork.FREEZE_RATE[] = FREEZE_RATE
MAGENetwork.BS[] = BATCH_SIZE
MAGENetwork.λ[] = REGULARIZATION
MAGENetwork.INTER_LOSSES[] = INTER_LOSSES
MAGENetwork.DP_RATE[] = DROPOUT_RATE
MAGENetwork.RESET_PB[] = RESET_PB
MAGENetwork.OPTIM[] = OPTIM
MAGENetwork.ACT[] = ACT
MAGENetwork.MAX[] = MAX

# MAGENetwork.OVERALLMODEL[] = MixedToScalarNN
const MASK::Bool =Parsed_args["mask"] == "true" # TODO

MAGENetwork.show_internal_params()

const _device = device!(DEVICE_CU)
@info _device
_device = CUDA.device()
@info _device

# Settings
try
    include("../utils/utils.jl")
    include("../utils/utils_aml.jl")
catch
end
try
    include("utils/utils.jl")
    include("utils/utils_aml.jl")
catch
end
#include(joinpath(home, "utils/utils.jl"))
disable_logging(Logging.Debug)

######################
# READ FILES #########
######################
annotations = DataFrame(readdlm(
        joinpath(dataset_base_path, annot_base_path, "annotations.dat")),
    ["path", "annot1", "annot2", "annot3"])

# Read Files
files = aml_files(annotations, _to_binary_atypical_label);
Y = extract_ys(files, get_label) # stratify according to TRUE label (detailed)

# Stratify 
val, test, train = stratifiedobs((collect(1:length(files)), Y), p=(0.3, 0.2)) # according to orig. paper 80% train, 20% test

# Re-label Ys (now that cells are stratified)
valy = _to_binary_atypical_label.(val[2])
trainy = _to_binary_atypical_label.(train[2]);
testy = _to_binary_atypical_label.(test[2]);
valy = map(x -> x == "TYPICAL" ? 1 : 2, valy)
trainy = map(x -> x == "TYPICAL" ? 1 : 2, trainy)
testy = map(x -> x == "TYPICAL" ? 1 : 2, testy)

trainy = [[x] for x in trainy]
valy = [[x] for x in valy]

trainx = get_color_channels.(files[collect(train[1])]);
valx = get_color_channels.(files[collect(val[1])]);
testx = get_color_channels.(files[collect(test[1])]);

# Add constants
# constants = [0.0, -1.0, 0.5, 2.0, 10, 20.0, 30.0]
# for container in (trainx, valx, testx)
#     for obs in container
#         push!(obs, constants...)
#     end
# end

tmp = countmap(trainy);
barplot(collect(keys(tmp)), collect(values(tmp)), title="Train")
tmp = countmap(valy);
barplot(collect(keys(tmp)), collect(values(tmp)), title="Validation")
tmp = countmap(testy);
barplot(collect(keys(tmp)), collect(values(tmp)), title="Test")

#
TRAINDataloader = RamDataLoader(trainx, trainy,
    collect(1:length(trainy)),
    ceil(Int, NSAMPLES / nt), NSAMPLES);
const VALDataloader = RamDataLoader(valx, valy,
    collect(1:length(valy)),
    ceil(Int, NSAMPLES / nt), NSAMPLES);

# NETWORK
sample_img = trainx[1][1]
Type2Dimg = typeof(sample_img)

# SKIMAGE 
@everywhere function float_caster2(n)
    if isnan(n) || isinf(n)
        return 0.0
    else
        clamp(convert(Float64, n), UTCGP.MIN_FLOAT[], UTCGP.MAX_FLOAT[])
        # clamp(convert(Float64, n), -1.0, 1.0)
    end
end
@everywhere @eval Type2Dimg = typeof($sample_img)
@everywhere @eval ret_0() = 0.0
if USE_SKI
    @info "Making SKI Function lib"
    @everywhere begin
        skimage_factories = [bundle_float_skimagemeasure]
        skimage_factories = [deepcopy(b) for b in skimage_factories]
        for factory_bundle in skimage_factories
            for (i, wrapper) in enumerate(factory_bundle)
                fn = wrapper.fn(Type2Dimg) # specialize
                wrapper.fallback = ret_0
                wrapper.caster = float_caster2
                factory_bundle.functions[i] =
                    UTCGP.FunctionWrapper(fn, wrapper.name, wrapper.caster, wrapper.fallback)
                specific_functions = typeof(fn).parameters[1]
                function (dp::ManualDispatcher{specific_functions})(inputs::Vararg{Any})
                    tt = typeof.(inputs)
                    fn = UTCGP._which_fn_in_manual_dispatcher(dp, tt)
                    # isnothing(fn) && throw(MethodError(dp, tt))
                    if isnothing(fn) || !(fn isa Function)
                        # msg::MethodError = MethodError(sum, tt)
                        msg = MethodError(dp, 1)
                        throw(msg)
                    else
                        fn_sure = identity(fn)
                        @fetch begin
                            fn_sure(inputs...)
                        end
                    end
                end
            end
        end
    end
end

fallback_intensity() = SImageND(IntensityPixel{N0f8}.(ones(N0f8, size(trainx[1][1]))))
fallback_binary() = SImageND(BinaryPixel{Bool}.(ones(Bool, size(trainx[1][1]))))
fallback_segment() = SImageND(SegmentPixel{Int}.(ones(Int, size(trainx[1][1]))))
image_intensity = UTCGP.get_image2Dintensity_factory_bundles();
image_binary = UTCGP.get_image2Dbinary_factory_bundles();
image_segment = UTCGP.get_image2Dsegment_factory_bundles();
Type2Dimg_intensity = typeof(fallback_intensity())
Type2Dimg_binary = typeof(fallback_binary())
Type2Dimg_segment = typeof(fallback_segment())
@show Type2Dimg_intensity Type2Dimg_binary Type2Dimg_segment

# push!(image2D, UTCGP.experimental_bundle_image2D_mask_factory) # TODO

for (factories, fallback, typeimg) in zip(
    [image_intensity, image_binary, image_segment],
    [fallback_intensity, fallback_binary, fallback_segment],
    [Type2Dimg_intensity, Type2Dimg_binary, Type2Dimg_segment],
)
    for factory_bundle in factories
        for (i, wrapper) in enumerate(factory_bundle)
            fn = wrapper.fn(typeimg) # specialize
            wrapper.fallback = fallback
            factory_bundle.functions[i] =
                UTCGP.FunctionWrapper(fn, wrapper.name, wrapper.caster, wrapper.fallback)
        end
    end
end

float_bundles = UTCGP.get_float_bundles()
# push!(float_bundles, bundle_float_imagegraph)

only_float_bundles = UTCGP.get_sr_float_bundles()

# glcm_b = deepcopy(UTCGP.experimental_bundle_float_glcm_factory) # TODO
# for (i, wrapper) in enumerate(glcm_b)
#     # try
#     fn = wrapper.fn(Type2Dimg) # specialize
#     glcm_b.functions[i] =
#         UTCGP.FunctionWrapper(fn, wrapper.name, float_caster2, ret_0)
# end
# push!(float_bundles, glcm_b)

if USE_SKI
    push!(float_bundles, skimage_factories...)
end

for factory_bundle in float_bundles
    for (i, wrapper) in enumerate(factory_bundle)
        wrapper.caster = float_caster2
    end
end
for factory_bundle in only_float_bundles
    for (i, wrapper) in enumerate(factory_bundle)
        wrapper.caster = float_caster2
    end
end

lib_image2D_intensity = Library(image_intensity);
lib_image2D_binary = Library(image_binary);
lib_image2D_segment = Library(image_segment);
lib_float = Library(float_bundles)
ml = MetaLibrary([lib_image2D_intensity, lib_image2D_binary, lib_image2D_segment, lib_float]);
ml_float = MetaLibrary([Library(only_float_bundles)]);

initial_pop = MNModel[]
n_elite = 5
for i in 1:n_elite
    mn_model = create_layers(Parsed_args, Type2Dimg_intensity, (Type2Dimg_binary, Type2Dimg_segment), ml, ml_float, N_NODES, NCLASSES)
    init_mn!(mn_model; size=:small)
    push!(initial_pop, mn_model)
end
initial_pop = MNPopulation(initial_pop);
for ind in initial_pop
    @info ind
end

rc = MNRunConf(
    ; gens=GENS, n_elite=n_elite, n_new=N_NEW, ts=TOUR_SIZE,
    mutation_n_models=MUTATION_N_MODELS,
    mutation_model=MUTATION_RATE
)

endpoint = PopVsSample(NCLASSES)

###########
# METRICS #
###########
folder = "metrics_mage"
folder = joinpath(folder, ID, string(SEED))
isdir(folder) || mkpath(folder)
checkpointer = checkpoint(1, folder)
f = open(joinpath(rootdir, folder, ID) * ".json", "w", lock=true)
metric_tracker = UTCGP.jsonTracker(Parsed_args, f)
train_tracker = jsonTrackerGA(metric_tracker,
    acc_callback([], [], "train"), "Train", nothing, nothing, 0.0, nothing); # will use the batch from the training loop to calc metrics
val_tracker = jsonTrackerGA(metric_tracker,
    acc_callback(valx, valy, "Val"), "Val", [], nothing, 0.0, checkpointer);


plateau_detector_to_activate_surrogates = PlateauDetector(1);
surrogate_explorer = SurrogateExplorer(plateau_detector_to_activate_surrogates;
    surrogate_epochs=EPOCHS_SURR, strategy=:loop, k=K, device=:gpu, variants_per_elite=1000);
epoch_callbacks = (train_tracker, val_tracker, surrogate_explorer);

# Add it to your existing epoch callbacks
epoch_callbacks = (train_tracker, val_tracker);

struct es <: UTCGP.AbstractCallable
    init_time
end
function (e::es)(t)
    global HLIMIT
    new_time = time()
    elapsed_s = (new_time - e.init_time)
    elapsed_m = elapsed_s / 60
    elapsed_h = elapsed_m / 60
    @info "TIME ELAPSED $elapsed_h"
    if elapsed_h > HLIMIT || elapsed_h + t > HLIMIT
        @info "Early Stopping because of time limit. Elapsed : $elapsed_h. Time gen :$t. Limit $(HLIMIT)"
        return true
    else
        return false
    end
end
es_callback = es(st)

pop, losses = fit_ga_network_batch(
    TRAINDataloader,
    nothing,
    N_REPETITIONS,
    initial_pop,
    rc,
    nothing,
    (surrogate_explorer,),
    # Callbacks before step (before looping through data)
    (mnga_pop_callback,),
    (mnga_mutation_callback!,),
    # output_mutation_callbacks::UTCGP.Mandatory_FN,
    (mnga_decoding_callback!,),
    # Callbacks per step (while looping through data)
    endpoint,
    nothing, #final_step_callbacks::UTCGP.Optional_FN,
    # Callbacks after step ::
    (mnga_selection_callback,),
    epoch_callbacks,
    (es_callback,), #early_stop_callbacks::UTCGP.Optional_FN,
    nothing #last_callback::UTCGP.Optional_FN,
)

UTCGP.save_json_tracker(metric_tracker)
checkpointer(0, pop)
et_script = time()
close(f)
println("TOTAL TIME IN SECONDS:$(et_script - st)")
println(val_tracker.best_loss * -1)

# Write result to file
# storage = MAGENetwork.STORAGE[]["joint_training"]
# Serialization.serialize(joinpath(OUTPUTDIR, "storage$(SEED).pickle"), storage)
result_file = joinpath(OUTPUTDIR, "result_$(SEED).txt")
open(result_file, "w") do io
    write(io, string(val_tracker.best_loss * -1))
    println("Writing results to $result_file : $(val_tracker.best_loss)")
end

# TRAIN MODEL
# examples = TRAINDataloader[1:(256*N_NN_BATCHES_TRAIN)];
# inputs_nn = [x[1] for x in examples];
# labels_nn = [x[2][1] for x in examples];
# program_data_nn = MAGENetwork.MAGEProgramInstance[]
# MAGEProgramInstance[]

# for (x, y) in zip(inputs_nn, labels_nn)
# push!(program_data_nn, MAGENetwork.MAGEProgramInstance(x, y))
# end

# val_examples = VALDataloader[1:(256*N_NN_BATCHES_VAL)];
# val_inputs_nn = [x[1] for x in val_examples];
# val_labels_nn = [x[2][1] for x in val_examples];
# val_program_data_nn = MAGENetwork.MAGEProgramInstance[]
# for (x, y) in zip(val_inputs_nn, val_labels_nn)
# push!(val_program_data_nn, MAGENetwork.MAGEProgramInstance(x, y))
# end
# m = initial_pop[1]
# MAGENetwork.with_device(model -> begin
#    MAGENetwork.train_model_alone(m, program_data_nn, val_program_data_nn; n_classes=2, max_epochs=20)
# end, m, :gpu)

# m = initial_pop[2]
# MAGENetwork.with_device(model -> begin
#    MAGENetwork.train_model_alone(m, program_data_nn, val_program_data_nn; n_classes=2, max_epochs=20)
# end, m, :gpu)

# function reset_random_weights_model!(model::MNModel)
#     for layer in model.mnsequence.mnlayers
#         for mage_and_surrogate in layer.programs
#             if mage_and_surrogate.surrogate isa MAGENetwork.NNSurrogateModel &&
#                mage_and_surrogate.surrogate.active
#                 mage_and_surrogate.surrogate.model = MAGENetwork.reinit_model(mage_and_surrogate.surrogate.model)
#                 @info "Reset"
#             end
#         end
#     end
# end
