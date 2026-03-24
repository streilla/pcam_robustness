@show ARGS
st = time()
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))

include(joinpath(home, "src", "mage_imports.jl"))
include(joinpath(home, "src", "mage_args.jl"))
include(joinpath(home, "src", "magenet_ski.jl"))
@add_arg_table s begin
    "--boost_K"
    arg_type = String
end
Parsed_args = parse_args(s)
@show Parsed_args

SURROGATE_INDEX = Parsed_args["boost_K"] # something like 1_5
@info "This model will be saved in $SURROGATE_INDEX"

const_vars = setup_constants(Parsed_args)
map(i -> eval(i), const_vars)
@info Parsed_args

USE_SKI ? addprocs(nt, exeflags = ["--threads=1"]) : nothing
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images

include_utils_and_disable_logging(home, false)
include(joinpath(home, "utils", "activations.jl"))

loss_type = Parsed_args["loss_type"]
T = nothing
if loss_type == "corr"
    T = IndVsSampleRegCorr
    @warn "Using Correlation fitness $T"
elseif loss_type == "rmse"
    T = IndVsSampleRegRMSE
    @warn "Using MSE fitness $T"
else
    exit()
end

######################
# READ FILES #########
######################
data_path = Parsed_args["data_location"]
nn_id = split(data_path, "/")[2]
surrogate_file = split(data_path, "/")[4]
K = match(r"[0-9]_[0-9]*", data_path).match
@info "Fitting module at $data_path. Module $K"

data = load(data_path)["single_stored_object"]
trainx, trainy = data.xs, data.ys
trainx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in trainx]
const TRAINDataloader = make_dataloader(trainx, trainy, NSAMPLES, nt)

val_data_path = Parsed_args["val_data_location"]
if val_data_path != ""
    data = load(val_data_path)["single_stored_object"]
    valx, valy = data.xs, data.ys
    valx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in valx]
    @assert length(valx) == length(valy)
    VALDataloader = make_dataloader(valx, valy, NSAMPLES, nt)
else
    valx, valy = nothing, nothing
    VALDataloader = nothing
end

@assert length(trainx) == length(trainy)
@info "Size train : $(length(trainx))"
@info "Size val : $(length(valx))"

const CLASSES = sort(unique(trainy))
const N_CLASSES = length(CLASSES)
const sample_img = trainx[1][1]

# Image Bundles
include(joinpath(home, "src", "magenet_image_bundles.jl"))
define_common_image_functions(sample_img)
skimage_factories = USE_SKI ? setup_skimage_distributed(Type2Dimg_binary) : nothing

# Float Bundles
float_bundles = UTCGP.get_float_bundles()
USE_SKI ? push!(float_bundles, skimage_factories...) : nothing
push!(float_bundles, bundle_float_imagegraph)
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
endpoint = PopVsSampleREG{T}()

###########
# METRICS #
##########
folder = OUTPUTDIR
folder = joinpath(folder, nn_id)
folder = joinpath(folder, "$(SURROGATE_INDEX[1])_$(SURROGATE_INDEX[3:end])")
isdir(folder) || mkpath(folder)
folder = joinpath(folder, "$(ID)__$(SEED)")
isdir(folder) || mkpath(folder)
checkpointer = checkpoint(5, folder)
f = open(joinpath(folder, "metrics_$SEED" * ".json"), "w", lock = true)
metric_tracker = UTCGP.jsonTracker(Parsed_args, f)

trackerx, trackery = trainx, trainy
if val_data_path != ""
    trackerx, trackery = valx, valy
end
val_tracker = jsonTrackerGA(
    metric_tracker,
    acc_callback(trackerx, trackery, "Val"), "Val", [], nothing,
    1.0, # worst loss
    checkpointer
);

es_limit = es(st)

@everywhere function gc_skimage(args...)
    return if isdefined(Main, :MAGE_SKIMAGE_MEASURE)
        @info "Running GC SKIMAGE MEASURE"
        MAGE_SKIMAGE_MEASURE.gc()
    end
end
function gc_skimage_callback(args...; kwargs...)
    return @everywhere gc_skimage()
end


best_genome, best_programs, gen_tracker = fit_ga_meanbatch_mt(
    TRAINDataloader,
    nothing,
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
    (val_tracker, gc_skimage_callback),
    # Final callbacks ?
    (es_limit,), #(budget_stop,), #(:default_early_stop_callback,), #
    nothing #repeat_metric_tracker # ..
)

Parsed_args["best_tracker_loss"] = val_tracker.best_loss

UTCGP.save_json_tracker(metric_tracker)
checkpointer(0, val_tracker.best_ind)
et_script = time()
close(f)
println("TOTAL TIME IN SECONDS:$(et_script - st)")
println(val_tracker.best_loss * -1)

# Write result to file
# storage = MAGENetwork.STORAGE[]["joint_training"]
# Serialization.serialize(joinpath(OUTPUTDIR, "storage$(SEED).pickle"), storage)
# result_file = joinpath(OUTPUTDIR, "result_$(SEED).txt")
# open(result_file, "w") do io
#     write(io, string(val_tracker.best_loss * -1))
#     println("Writing results to $result_file : $(val_tracker.best_loss)")
# end


# test prog
# payload = deserialize("surrogates/1_1/checkpoint_0.pickle")
# best_ind2 = payload["best_genome"]
# best_ind2 = best_ind
# new_pop = [best_ind2]
# progs = UTCGP.PopulationPrograms([UTCGP.decode_with_output_nodes(i, ml, model_arch, shared_in) for i in new_pop])
# perf = [1.0 for i in new_pop]
# val_tracker(
#     perf,
#     Population(new_pop),
#     1,
#     run_conf,
#     model_arch,
#     node_config,
#     ml,
#     shared_in,
#     progs,
#     [0.1],
#     progs.population_programs,
#     [1],
#     view(trainx, :)
# )

# # Get all outputs from a prog
# best_ind = val_tracker.best_ind
# xs = trainx
# ys = trainy
# prog = UTCGP.decode_with_output_nodes(best_ind, ml, model_arch, shared_in)
# pop_of_one = UTCGP.PopulationPrograms([prog])
# OUTS = []
# for (sample_idx, (x, y)) in enumerate(zip(xs, ys))
#     UTCGP.reset_programs!.(pop_of_one)
#     UTCGP.replace_shared_inputs!(
#         pop_of_one,
#         x,
#     )
#     outputs, times = UTCGP.evaluate_population_programs_with_time(
#         pop_of_one,
#         model_arch,
#         ml,
#     )
#     outputs = MAGENetwork.INTER_ACT.(outputs)
#     push!(OUTS, (outputs[1][1], y))
# end

# res = [i[2] - i[1] for i in OUTS]
# trainy = res
# TRAINDataloader2 = make_dataloader(trainx, trainy, NSAMPLES, nt)
# best_genome, best_programs, gen_tracker = fit_ga_meanbatch_mt(
#     # @enter fit_ga(
#     TRAINDataloader2,
#     nothing,
#     :reg,
#     shared_in,
#     initial_pop,
#     model_arch,
#     node_config,
#     run_conf,
#     ml,
#     # Callbacks before training
#     nothing,
#     # Callbacks before step
#     (:ga_population_callback,),
#     (:ga_numbered_new_material_mutation_callback,),
#     (:ga_output_mutation_callback,),
#     (:default_decoding_callback,),
#     # Endpoints
#     endpoint,
#     # STEP CALLBACK
#     nothing,
#     # Callbacks after step
#     (:ga_elite_selection_callback,),
#     # Epoch Callback
#     (val_tracker,), #[metric_tracker, test_tracker, sn_writer_callback],
#     # Final callbacks ?
#     nothing, #(budget_stop,), #(:default_early_stop_callback,), #
#     nothing #repeat_metric_tracker # ..
# )
