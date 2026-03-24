st = time()
ENV["JULIA_DEBUG"] = "UTCGP"
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))

include(joinpath(home, "src", "mage_imports.jl"))
include(joinpath(home, "src", "mage_args.jl"))
include(joinpath(home, "src", "magenet_ski.jl"))

Parsed_args["data_location"] = "surrogates_cifar/_EXP1/nn_surrogates/surrogate_1_1.jld2"
Parsed_args["trial_id"] = "_EXPX"
Parsed_args["output_dir"] = "surrogates_cifar"

const_vars = setup_constants(Parsed_args)
map(i -> eval(i), const_vars)
@info Parsed_args

USE_SKI ? addprocs(nt, exeflags = ["--threads=1"]) : nothing
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images
#pinthreads(:cores) ; println(threadinfo(; slurm = true, color = false))

include_utils_and_disable_logging(home, false)

######################
# READ FILES #########
######################
data_path = Parsed_args["data_location"]
nn_id = split(data_path, "/")[2]
surrogate_file = split(data_path, "/")[4]
K = match(r"[0-9]_[0-9]*", data_path).match
@info "Fitting module at $data_path. Module $K"
data = load(data_path)["single_stored_object"]

trainx, trainy = data.xs, data.model_preds
true_labels = data.gt

const TRAINDataloader = make_dataloader(trainx, trainy, NSAMPLES, nt)
# const VALDataloader = make_dataloader(valx, valy, NSAMPLES, nt)

const N_CLASSES = sort(unique(true_labels))
const sample_img = trainx[1][1]
define_common_image_functions(sample_img)
skimage_factories = USE_SKI ? setup_skimage_distributed(nt, Type2Dimg) : nothing

# Image Bundles
include(joinpath(home, "src", "magenet_image_bundles.jl"))

# Float Bundles
float_bundles = UTCGP.get_float_bundles()
USE_SKI ? push!(float_bundles, skimage_factories...) : nothing
# push!(float_bundles, bundle_float_imagegraph) # TODO
glcm_b = deepcopy(UTCGP.experimental_bundle_float_glcm_factory)
push!(float_bundles, glcm_b)

set_bundle_casters!(float_bundles, float_caster2)

# Metalibs
ml = ml_from_vbundles([image_intensity, image_binary, image_segment, float_bundles])

n_ins = trainx[1] |> length
model_arch = modelArchitecture( # TODO
    [Type2Dimg_intensity for i in 1:n_ins],
    [1 for i in 1:n_ins],
    [Type2Dimg_intensity, Type2Dimg_binary, Type2Dimg_segment, Float64],
    [Float64],
    [4]
)
node_config = nodeConfig(N_NODES, 1, 3, n_ins)

initial_pop = UTGenome[]
shared_in = nothing
for i in 1:N_ELITE
    global shared_in
    shared_inputs, ut_genome = make_evolvable_utgenome(
        model_arch, ml, node_config
    )
    initialize_genome!(ut_genome)
    correct_all_nodes!(ut_genome, model_arch, ml, shared_inputs)
    fix_all_output_nodes!(ut_genome)
    push!(initial_pop, ut_genome)
    shared_in = shared_inputs
end

run_conf = RunConfGA(
    N_ELITE, N_NEW, TOUR_SIZE, MUTATION_RATE + 0.1, 0.1, GENS
)

# ENDPOINT ---
endpoint = PopVsSampleReg()

###########
# METRICS #
##########
folder = OUTPUTDIR
folder = joinpath(folder, nn_id)
folder = joinpath(folder, "$(K[1])_$(K[3:end])")
isdir(folder) || mkpath(folder)
folder = joinpath(folder, "$(ID)__$(SEED)")
isdir(folder) || mkpath(folder)
checkpointer = checkpoint(100, folder)
f = open("precompile_out.json", "w", lock = true)
metric_tracker = UTCGP.jsonTracker(Parsed_args, f)
val_tracker = jsonTrackerGA(
    metric_tracker,
    acc_callback(trainx, trainy, "Val"), "Val", [], nothing,
    1.0, # worst loss
    checkpointer
);

es_limit = es(st)

best_genome, best_programs, gen_tracker = fit_ga_meanbatch_mt(
    # @enter fit_ga(
    TRAINDataloader,
    nothing,
    :reg,
    shared_in,
    initial_pop,
    model_arch,
    node_config,
    run_conf,
    ml,
    # Callbacks before training
    nothing,
    # Callbacks before step
    (:ga_population_callback,),
    (:ga_numbered_new_material_mutation_callback,),
    (:ga_output_mutation_callback,),
    (:default_decoding_callback,),
    # Endpoints
    endpoint,
    # STEP CALLBACK
    nothing,
    # Callbacks after step
    (:ga_elite_selection_callback,),
    # Epoch Callback
    (val_tracker,), #[metric_tracker, test_tracker, sn_writer_callback],
    # Final callbacks ?
    (es_limit,), #(budget_stop,), #(:default_early_stop_callback,), #
    nothing #repeat_metric_tracker # ..
)
