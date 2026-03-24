s = ArgParseSettings()
@add_arg_table s begin
    "--seed"
    arg_type = Int
    default = 42
    "--data_location"
    arg_type = String
    default = ""
    "--val_data_location"
    arg_type = String
    default = ""
    "--output_dir"
    arg_type = String
    default = "surrogates"
    "--trial_id"
    arg_type = String
    default = "_1"
    "--mutation_rate"
    arg_type = Int
    default = 1
    "--mutation_n_models"
    arg_type = Int
    default = 2
    "--n_nodes"
    arg_type = Int
    default = 30
    "--n_new"
    arg_type = Int
    default = 100
    "--n_elite"
    arg_type = Int
    default = 2
    "--tour_size"
    arg_type = Int
    default = 100
    "--n_samples"
    arg_type = Int
    default = 200
    "--n_repetitions"
    arg_type = Int
    default = 1
    "--err_w"
    arg_type = Float64
    default = 0.8
    "--time_w"
    arg_type = Float64
    default = 100.0
    # MAGE FITTING
    "--generations_mage"
    arg_type = Int
    default = 40
    "--lambda_mage"
    arg_type = Int
    default = 20
    "--trainsize_mage"
    arg_type = Int
    default = 10000
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
    default = 3
    "--n_nn_batches_train"
    arg_type = Int
    default = 100
    "--n_nn_batches_val"
    arg_type = Int
    default = 70
    # architecture
    "--n_layers"
    arg_type = Int
    default = 2
    "--n_layers_img"
    arg_type = Int
    default = 1
    "--l1_size"
    arg_type = Int
    default = 4
    "--l1_out_size"
    arg_type = Int
    default = 1
    "--l2_size"
    arg_type = Int
    default = 8
    "--l2_out_size"
    arg_type = Int
    default = 5
    "--l3_size"
    arg_type = Int
    default = 15
    "--l4_size"
    arg_type = Int
    default = 2
    # ALIGN
    "--th_for_align" # MAGE will try to fit if initial loss > than this
    arg_type = Float64
    default = 0.1
    "--th_for_es"
    arg_type = Float64
    default = 100.0
    "--reset_pb"
    arg_type = Float64
    default = 0.01
    "--dropout_rate"
    arg_type = Float64
    default = 0.2
    "--freeze_rate"
    arg_type = Float64
    default = 0.2
    "--batch_size"
    arg_type = Int
    default = 128
    "--regularization"
    arg_type = Float32
    default = 0.0001
    "--inter_losses"
    arg_type = Float32
    default = 0.1f0
    "--mask"
    arg_type = String
    default = "false"
    "--optim"
    arg_type = String
    default = "adam"
    "--act"
    arg_type = String
    default = "relu"
    "--max"
    arg_type = Int
    default = 1
    # LIBS
    "--use_ski"
    arg_type = String
    default = "false"
end
rootdir = "./"


function setup_constants(args)
    return (
        :(const USE_SKI = $args["use_ski"] == "true"),
        :(const SEED::Int = $args["seed"]),
        :(Random.seed!(SEED)),
        :(const ERR_WEIGHT::Float64 = $args["err_w"]),
        :(const NERR_WEIGHT::Float64 = 1.0 - ERR_WEIGHT),
        :(const TIMEPENALTY::Int = $args["time_w"]),
        :(const N_REPETITIONS = $args["n_repetitions"]),
        :(const NSAMPLES::Int = $args["n_samples"]),
        :(const GENS::Int = 100),
        :(const HLIMIT::Float64 = 8),
        :(const N_NN_BATCHES_TRAIN::Int = $args["n_nn_batches_train"]),
        :(const N_NN_BATCHES_VAL::Int = $args["n_nn_batches_val"]),
        :(const DEVICE_CU::Int = $args["device"]),
        :(const K::Int = $args["n_surr"]),
        :(const EPOCHS_SURR::Int = $args["epochs_surr"]),
        :(const N_LAYERS::Int = $args["n_layers"]),
        :(const N_LAYERS_IMG::Int = $args["n_layers_img"]),
        :(const L1_SIZE::Int = $args["l1_size"]),
        :(const L1_OUT_SIZE::Int = $args["l1_out_size"]),
        :(const L2_SIZE::Int = $args["l2_size"]),
        :(const L2_OUT_SIZE::Int = $args["l2_out_size"]),
        :(const L3_SIZE::Int = $args["l3_size"]),
        :(const L4_SIZE::Int = $args["l4_size"]),
        :(const GENERATIONS_MAGE::Int = $args["generations_mage"]),
        :(const LAMBDA_MAGE::Int = $args["lambda_mage"]),
        :(const TRAINSIZE_MAGE::Int = $args["trainsize_mage"]),
        :(const MUTATION_RATE::Int = $args["mutation_rate"]),
        :(const MUTATION_N_MODELS::Int = $args["mutation_n_models"]),
        :(const N_NODES::Int = $args["n_nodes"]),
        :(const N_NEW::Int = $args["n_new"]),
        :(const N_ELITE::Int = $args["n_elite"]),
        :(const TOUR_SIZE::Int = $args["tour_size"]),
        :(const TH_FOR_ALIGN::Float64 = $args["th_for_align"]),
        :(const TH_FOR_ES::Float64 = $args["th_for_es"]),
        :(const RESET_PB::Float64 = $args["reset_pb"]),
        :(const DROPOUT_RATE::Float64 = $args["dropout_rate"]),
        :(const FREEZE_RATE::Float64 = $args["freeze_rate"]),
        :(const BATCH_SIZE::Int = $args["batch_size"]),
        :(const REGULARIZATION::Float32 = $args["regularization"]),
        :(const INTER_LOSSES::Float32 = $args["inter_losses"]),
        :(const MASK::Bool = $args["mask"] == "true"),
        :(const OPTIM::String = $args["optim"]),
        :(const ACT::Bool = $args["act"] == "tanh"),
        :(const MAX::Int = $args["max"]),
        :(const OUTPUTDIR::String = $args["output_dir"]),
        :(const ID::String = $args["trial_id"]),
        :(STO_WARMUP::Int = 2),
        :(STO_COOLDOWN::Int = 15),
        :(const NODESLAST::Int = 300),
        :(const W::Int = 32),
        :(const H::Int = W),
        :(const FOLLOW = Ref{Bool}(true)),
        :(rootdir = pwd()),
        :(const MASK::Bool = $Parsed_args["mask"] == "true"),
    ) # TODO
end

function setup_cuda_device(DEVICE_CU)
    _device = device!(DEVICE_CU)
    @info _device
    _device2 = CUDA.device()
    @info _device2
    return _device2
end

function include_utils_and_disable_logging(home)
    include(joinpath(home, "utils", "utils.jl"))
    include(joinpath(home, "utils", "utils_aml.jl"))
    include(joinpath(home, "utils", "datasets.jl"))
    include(joinpath(home, "utils", "utils_magenetwork.jl"))
    return disable_logging(Logging.Debug)
end
