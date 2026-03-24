s = ArgParseSettings()
@add_arg_table s begin
    "--seed"
    arg_type = Int
    default = 1
    "--data_location"
    arg_type = String
    default = ""
    "--val_data_location"
    arg_type = String
    default = ""
    "--output_dir"
    arg_type = String
    default = "metrics_mage"
    "--trial_id"
    arg_type = String
    default = ""
    "--gens"
    arg_type = Int
    default = 10
    "--mutation_rate"
    arg_type = Int
    default = 1
    "--n_nodes"
    arg_type = Int
    default = 100
    "--n_new"
    arg_type = Int
    default = 128
    "--n_elite"
    arg_type = Int
    default = 10
    "--tour_size"
    arg_type = Int
    default = 8
    "--n_samples"
    arg_type = Int
    default = 200
    "--err_w"
    arg_type = Float64
    default = 0.8
    "--time_w"
    arg_type = Float64
    default = 100.0

    # BUDGET TIME
    "--time"
    arg_type = Int
    default = 2

    # ACT
    "--act"
    arg_type = String
    default = "identity"

    # APPLICABLE ONLY TO FIT
    "--loss_type"
    arg_type = String

    # APPLICABLE ONLY TO FIT
    "--data_aug"
    arg_type = Bool
    default = false

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
        :(const NSAMPLES::Int = $args["n_samples"]),
        :(const GENS::Int = $args["gens"]),
        :(const HLIMIT::Float64 = $args["time"]),
        :(const MUTATION_RATE::Int = $args["mutation_rate"]),
        :(const N_NODES::Int = $args["n_nodes"]),
        :(const N_NEW::Int = $args["n_new"]),
        :(const N_ELITE::Int = $args["n_elite"]),
        :(const TOUR_SIZE::Int = $args["tour_size"]),
        :(const OUTPUTDIR::String = $args["output_dir"]),
        :(const ID::String = $args["trial_id"]),
        :(const W::Int = 32),
        :(const H::Int = W),
        :(rootdir = pwd()),
    ) # TODO
end

function include_utils_and_disable_logging(home, magenet::Bool = false)
    include(joinpath(home, "utils", "utils.jl"))
    include(joinpath(home, "utils", "utils_aml.jl"))
    include(joinpath(home, "utils", "datasets.jl"))
    if magenet
        include(joinpath(home, "utils", "utils_magenetwork.jl"))
    end
    return disable_logging(Logging.Debug)
end
