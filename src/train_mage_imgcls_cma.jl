using MAGE_PYCMA
st = time()
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))

include(joinpath(home, "src", "mage_imports.jl"))
include(joinpath(home, "src", "mage_args.jl"))
include(joinpath(home, "src", "magenet_ski.jl"))
Parsed_args = parse_args(s)
@show Parsed_args

const_vars = setup_constants(Parsed_args)
map(i -> eval(i), const_vars)

USE_SKI ? addprocs(clamp(nt, 3, 10), exeflags = ["--threads=1", "--heap-size-hint=1GB"]) : nothing
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images
# pinthreads(:cores) ; println(threadinfo(; slurm = true, color = false))

include_utils_and_disable_logging(home, false)
include(joinpath(home, "utils", "activations.jl"))

to_augment = Parsed_args["data_aug"]
if to_augment
    include(joinpath(home, "utils", "noise.jl"))
    function Main.sample(d::RamDataLoader, how_much::Int)
        idx = sample(d.indices, how_much)
        xs = d.xs[idx]
        ys = d.ys[idx]

        # add brigthness noise
        PIXEL_TYPE = eltype(xs[1][1])
        SIZE = UTCGP._get_image_tuple_size(xs[1][1])
        brightness_values = collect(-0.3:0.05:0.3)
        new_xs = similar(xs)
        for (i, x) in enumerate(xs)
            new_x = similar(x)
            z = add_brightness(map(y -> float64.(y.img), x), rand(brightness_values))
            new_x .= map(y -> SImageND(reinterpret.(PIXEL_TYPE, y), SIZE), z)
            new_xs[i] = new_x
        end
        return collect(zip(new_xs, ys)), idx
    end
end

######################
# READ FILES #########
######################
data_path = Parsed_args["data_location"]
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
push!(float_bundles, bundle_float_imagegraph) # TODO
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
    [Float64 for i in 1:N_CLASSES],
    [4 for i in 1:N_CLASSES]
)
node_config = nodeConfig(N_NODES, 1, 3, n_ins)

initial_pop = UTGenome[]
shared_in = nothing
for i in 1:(N_NEW + N_ELITE)
    global shared_in
    shared_inputs, ut_genome = make_evolvable_utgenome(
        model_arch, ml, node_config
    )
    initialize_genome!(ut_genome)
    correct_all_nodes!(ut_genome, model_arch, ml, shared_inputs)
    fix_all_output_nodes!(ut_genome)
    make_cma_nodes!(ut_genome, 20, 4)
    push!(initial_pop, ut_genome)
    shared_in = shared_inputs
end

run_conf = RunConfGA(
    N_ELITE, N_NEW, TOUR_SIZE, MUTATION_RATE + 0.1, 0.1, GENS
)

# ENDPOINT ---
endpoint = PopVsSampleCLS{IndVsSample}(N_CLASSES)

###########
# METRICS #
##########
folder = Parsed_args["output_dir"]
folder = joinpath(folder, Parsed_args["trial_id"])
folder = joinpath(folder, "mage_imgcls")
isdir(folder) || mkpath(folder)
folder = joinpath(folder, "$(ID)__$(SEED)")
isdir(folder) || mkpath(folder)
checkpointer = checkpoint(5, folder)
f = open(joinpath(folder, "metrics_$SEED" * ".json"), "w", lock = true)
metric_tracker = UTCGP.jsonTracker(Parsed_args, f)
val_tracker = jsonTrackerGA(
    metric_tracker,
    acc_callback(valx, valy, "Val"), "Val", [], nothing,
    1.0, # worst loss
    checkpointer
);

es_limit = es(st)

#overload inter_act
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
    (val_tracker, gc_skimage_callback), #[metric_tracker, test_tracker, sn_writer_callback],
    # Final callbacks ?
    nothing, #(es_limit,), #(budget_stop,), #(:default_early_stop_callback,), #
    nothing #repeat_metric_tracker # ..
    ; use_cma = true, cma_at = 4
)

Parsed_args["best_tracker_loss"] = val_tracker.best_loss

UTCGP.save_json_tracker(metric_tracker)
checkpointer(0, val_tracker.best_ind)
et_script = time()
close(f)
println("TOTAL TIME IN SECONDS:$(et_script - st)")
println(val_tracker.best_loss * -1)
