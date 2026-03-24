using ArgParse
using Serialization
using DataFramesMeta
using DataFrames
using HDF5

st = time()
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(file)
#home *= "/pcam_robustness"

s = ArgParseSettings()
@add_arg_table s begin
    "--model_path"
    arg_type = String # where to read the data to make a features dataset by running the models
    "--data_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    default = ""
    "--labels_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    default = ""
    "--val_data_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    default = ""
    "--val_labels_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    default = ""
    "--test_data_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    default = ""
    "--test_labels_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    default = ""
    "--trial_id"
    arg_type = String #
    "--output_dir"
    arg_type = String
    "--act"
    arg_type = String
    default = "identity"
    "--use_ski"
    arg_type = Bool
    default = false
    "--noise_level"
    arg_type = String
end
rootdir = "./"
Parsed_args = parse_args(s)
@show Parsed_args

include(joinpath(home, "src", "mage_imports.jl"))
include(joinpath(home, "src", "magenet_ski.jl"))

NOISE_LEVEL = Parsed_args["noise_level"]
USE_SKI = Parsed_args["use_ski"]
USE_SKI ? addprocs(min(nt, 15), exeflags = ["--threads=1", "--heap-size-hint=1GB"]) : nothing
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images

include(joinpath(home, "utils", "utils.jl"))
include(joinpath(home, "utils", "utils_aml.jl"))
include(joinpath(home, "utils", "datasets.jl"))
include(joinpath(home, "utils", "activations.jl"))

# Read data ---
model_path = Parsed_args["model_path"]
train_location = Parsed_args["data_location"]
has_train_data = train_location != ""
val_location = Parsed_args["val_data_location"]
has_val_data = val_location != ""
test_location = Parsed_args["test_data_location"]
has_test_data = test_location != ""
val_labels_location = Parsed_args["val_labels_location"]
test_labels_location = Parsed_args["test_labels_location"]

trainx, trainy = nothing, nothing
valx, valy = nothing, nothing
testx, testy = nothing, nothing

# -- load train
if has_train_data
    data_path = Parsed_args["data_location"]
    @info "Reading data at $data_path."
    #data = load(data_path)["single_stored_object"]
    data = read(data_path)
    trainx, trainy = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in data.xs], data.ys
    @assert length(trainx) == length(trainy)
    @info "Samples in dataset(s): $(length(trainx))"
    n_classes = sort(unique(trainy))
    sample_img_ = trainx[1]
end

# -- load val
corse_labels_val = []
if has_val_data
    @info "Reading data at $val_location."
    #val_data = load(val_location)["single_stored_object"]
    h5file_data = h5open(val_location)
    datasets = HDF5.get_datasets(h5file_data)
    val_data = datasets[1]
    data = val_data[]
    h5file_labels = h5open(val_labels_location)
    datasets_labels = HDF5.get_datasets(h5file_labels)
    val_labels = datasets_labels[1]
    labels = val_labels[]
    valx =  [[SImageND(reinterpret.(IntensityPixel{N0f8}, data[c,:,:,img]')) for c in 1:3] for img in 1:size(data,4)]
    valy = vcat(labels...) + [1 for _ in 1:size(labels, 4)]
    #sample_img_ = valx[1]
    @assert length(valx) == length(valy)
    @info "Samples in dataset(s): $(length(valx))"
    n_classes = sort(unique(valy))
    @info "N classes : $n_classes"
end


corse_labels_test = []
if has_test_data
    @info "Reading data at $test_location."
    #test_data = load(test_location)["single_stored_object"]
    #testx, testy = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in test_data.xs], test_data.ys
    h5file_data = h5open(test_location)
    datasets = HDF5.get_datasets(h5file_data)
    test_data = datasets[1]
    data = test_data[]
    h5file_labels = h5open(test_labels_location)
    datasets_labels = HDF5.get_datasets(h5file_labels)
    test_labels = datasets_labels[1]
    labels = test_labels[]
    testx =  [[SImageND(reinterpret.(IntensityPixel{N0f8}, data[c,:,:,img]')) for c in 1:3] for img in 1:size(data,4)]
    testy = vcat(labels...) + [1 for _ in 1:size(labels, 4)] #Because Julia is 1-indexed
    @assert length(testx) == length(testy)
    @info "Samples in dataset(s): $(length(testx))"
    n_classes = sort(unique(testy))
    sample_img_ = testx[1]

    # if haskey(test_data, :extras) && haskey(test_data.extras, :metadata)
    #     @info "Reading Corse TEST labels"
    #     push!(corse_labels_test, map(x -> x.folder_name, test_data.extras.metadata)...)
    # end
end

# Setup libraries
const N_CLASSES = n_classes |> length
const sample_img = sample_img_[1]
define_common_image_functions(sample_img)
# Image Bundles
include(joinpath(home, "src", "magenet_image_bundles.jl"))

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
n_ins = length(sample_img_)

model_arch = modelArchitecture(
    [Type2Dimg_intensity for i in 1:n_ins],
    [1 for i in 1:n_ins],
    [Type2Dimg_intensity, Type2Dimg_binary, Type2Dimg_segment, Float64],
    [Float64 for i in 1:N_CLASSES],
    [4 for i in 1:N_CLASSES]
)

# Setup outputs
# folder = "mage_results"
# isdir(folder) || mkdir(folder)
# folder = joinpath(folder, Parsed_args["output_dir"])
folder = Parsed_args["output_dir"]
isdir(folder) || mkdir(folder)
folder = joinpath(folder, Parsed_args["trial_id"])
isdir(folder) || mkdir(folder)

ind = Parsed_args["model_path"]
payload = deserialize(ind)
ind_decoded = payload["best_genome"]

N_NODES = ind_decoded[1] |> length
node_config = nodeConfig(N_NODES, 1, 3, n_ins)
shared_in, _ = make_evolvable_utgenome(
    model_arch, ml, node_config
)

prog = UTCGP.decode_with_output_nodes(ind_decoded, ml, model_arch, shared_in)

# Running
nt = Threads.nthreads()
pop = [ind_decoded]
function run_pop_on_data(pop, x, y)
    OUTS_ALL = []
    for (i, ind) in enumerate(pop)
        @info "Running $i"
        local K = (1, i)
        best_ind = ind
        xs = x
        ys = y
        prog = UTCGP.decode_with_output_nodes(best_ind, ml, model_arch, shared_in)
        progs = [deepcopy(prog) for i in 1:nt]
        pop_size = length(progs)
        n_samples = length(xs)
        thread_size = ceil(Int, n_samples / pop_size)
        pop_subsets = Iterators.partition(1:n_samples, thread_size)
        OUTS = Vector{NamedTuple}(undef, n_samples)
        tasks = []
        for (idx, pop_subset) in enumerate(pop_subsets)
            t = Threads.@spawn begin
                xs_v, ys_v = xs[pop_subset], ys[pop_subset]
                out_v = @view OUTS[pop_subset]
                prog_copy = deepcopy(progs[idx])
                @assert length(out_v) == length(xs_v) == length(ys_v)
                for (sample_idx, (x, y)) in enumerate(zip(xs_v, ys_v))
                    UTCGP.reset_program!.(prog_copy)
                    UTCGP.replace_shared_inputs!(prog_copy, x)
                    outputs = UTCGP.evaluate_individual_programs(prog_copy, model_arch.chromosomes_types, ml)
                    outputs = INTER_ACT[].(outputs)
                    out_v[sample_idx] = (pred = argmax(outputs), gt = y)
                end
            end
            push!(tasks, t)
        end
        fetch.(tasks)
        push!(OUTS_ALL, deepcopy(OUTS))
    end
    return OUTS_ALL
end

function calc_bacc(obs_and_preds)
    gt = [i.gt for i in obs_and_preds]
    preds = [i.pred for i in obs_and_preds]
    return StatisticalMeasures.BalancedAccuracy()(preds, gt)
end

function get_noise_level(level::String)
    if level == ""
        @info "matches empty string"
        return 0.0
    end
    if !isnothing(match(r"neg|pos", level))
        @info "matches neg or pos"
        m = match(r"0_[0-9]*", level)
        matched = m.match
        matched = replace(matched, "_" => ".")
        nb = parse(Float64, matched)
        if occursin("neg", level)
            return nb * -1
        else
            return nb
        end
    end
    if !isnothing(match(r"_", level))
        @info "matches _"
        m = match(r"0_[0-9]*", level)
        if isnothing(m)
            return 0.0
        else
            matched = m.match
            matched = replace(matched, "_" => ".")
            return parse(Float64, matched)
        end
    else
        return parse(Float64, level) 
    end
end

if has_train_data
    noise_level = get_noise_level(NOISE_LEVEL)
    outs_train = run_pop_on_data(pop, trainx, trainy)[1]
    train_bacc = calc_bacc(outs_train)
    @show train_bacc noise_level
    open(joinpath(folder, "noise_stats.txt"), "a") do io
        write(io, "$model_path;$train_location;$noise_level;$train_bacc\n")
    end
end

if has_val_data
    noise_level = get_noise_level(NOISE_LEVEL)
    outs_val = run_pop_on_data(pop, valx, valy)[1]
    val_bacc = calc_bacc(outs_val)
    @show val_bacc noise_level
    open(joinpath(folder, "noise_stats.txt"), "a") do io
        write(io, "$model_path;$val_location;$noise_level;$val_bacc\n")
    end
    # SLIDE
    if !isempty(corse_labels_val)
        @info "Writing data for VAL in slide level"
        df_slide = DataFrame([(folder = folder, pred = preds_truth.pred, gt = preds_truth.gt) for (folder, preds_truth) in zip(corse_labels_val, outs_val)])
        preds_at_slide_level = @chain df_slide begin
            @groupby :folder
            @combine :mean_pred = mean(:pred) :gt = unique(:gt)
            @transform :avg_pred = round.(Int, :mean_pred)
        end
        slide_val_bacc = calc_bacc([(pred = row.avg_pred, gt = row.gt) for row in eachrow(preds_at_slide_level)])
        @show "VAL" slide_val_bacc noise_level
        open(joinpath(folder, "noise_stats_slide.txt"), "a") do io
            write(io, "$model_path;$val_location;$noise_level;$slide_val_bacc\n")
        end
    end
end

if has_test_data
    noise_level = get_noise_level(NOISE_LEVEL)
    outs_test = run_pop_on_data(pop, testx, testy)[1]
    test_bacc = calc_bacc(outs_test)
    @show test_bacc noise_level
    open(joinpath(folder, "noise_stats.txt"), "a") do io
        write(io, "$model_path;$test_location;$noise_level;$test_bacc\n")
    end
    if !isempty(corse_labels_test)
        @info "Writing data for test in slide level"
        df_slide = DataFrame([(folder = folder, pred = preds_truth.pred, gt = preds_truth.gt) for (folder, preds_truth) in zip(corse_labels_test, outs_test)])
        preds_at_slide_level = @chain df_slide begin
            @groupby :folder
            @combine :mean_pred = mean(:pred) :gt = unique(:gt)
            @transform :avg_pred = round.(Int, :mean_pred)
        end
        slide_test_bacc = calc_bacc([(pred = row.avg_pred, gt = row.gt) for row in eachrow(preds_at_slide_level)])
        @show "TEST" slide_test_bacc noise_level
        open(joinpath(folder, "noise_stats_slide.txt"), "a") do io
            write(io, "$model_path;$test_location;$noise_level;$slide_test_bacc\n")
        end
    end

end
