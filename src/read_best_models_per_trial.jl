using ArgParse
st = time()
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))
s = ArgParseSettings()
@add_arg_table s begin
    "--data_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    "--val_data_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    default = ""
    "--trial_id"
    arg_type = String #
    "--use_only_bests"
    arg_type = Bool #
    default = true
    "--output_dir"
    arg_type = String
    "--use_ski"
    arg_type = String
    default = "false"
    "--act"
    arg_type = String
    default = "identity"
end
rootdir = "./"
Parsed_args = parse_args(s)
@show Parsed_args

include(joinpath(home, "src", "mage_imports.jl"))
include(joinpath(home, "src", "magenet_ski.jl"))

USE_SKI = Parsed_args["use_ski"]
USE_SKI ? addprocs(nt, exeflags = ["--threads=1"]) : nothing
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images

include(joinpath(home, "utils", "utils.jl"))
include(joinpath(home, "utils", "utils_aml.jl"))
include(joinpath(home, "utils", "datasets.jl"))
include(joinpath(home, "utils", "activations.jl"))

use_bests = Parsed_args["use_only_bests"]

# Read data ---
data_path = Parsed_args["data_location"]
data = load(data_path)["single_stored_object"]
trainx, trainy = data.xs, data.ys
trainx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in trainx]

val_location = Parsed_args["val_data_location"]
has_val_data = val_location != ""
if has_val_data
    data = load(val_location)["single_stored_object"]
    valx, valy = data.xs, data.ys
    valx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in valx]
    @assert length(valx) == length(valy)
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
n_ins = length(trainx[1])

model_arch = modelArchitecture( # TODO
    [Type2Dimg_intensity for i in 1:n_ins],
    [1 for i in 1:n_ins],
    [Type2Dimg_intensity, Type2Dimg_binary, Type2Dimg_segment, Float64],
    [Float64],
    [4]
)

folder = Parsed_args["output_dir"]
folder = joinpath(folder, Parsed_args["trial_id"])
folders_to_read = filter(x -> occursin(r"[0-9]_[0-9]*", x), readdir(folder))
folders_to_read = [joinpath(folder, x) for x in folders_to_read]

Bests = Dict()
Files_Losses = Dict()
for folder_to_read in folders_to_read
    @info "Reading $folder_to_read"
    best_losses = []
    for (root, dirs, files) in walkdir(folder_to_read)
        if isempty(files)
            @show root
            continue
        end
        @show files
        metrics_file = filter(x -> occursin("metrics", x), files)[1]
        metrics_file_content = readlines(joinpath(root, metrics_file))
        @info "Reading : $metrics_file"
        try
            metrics = JSON.parse(metrics_file_content[end])
            best_f = metrics["params"]["best_tracker_loss"]
            push!(
                best_losses,
                (loss = best_f, root = root, metrics_file = metrics_file)
            )
        catch e
            @show e
            @warn root
        end
    end
    sort!(best_losses, by = first)
    @show best_losses

    if use_bests
        best = best_losses[begin]
        if length(best_losses) > 1
            others_min = map(x -> x.loss, best_losses[(begin + 1):end]) |> minimum
            others_mean = map(x -> x.loss, best_losses[(begin + 1):end]) |> mean
            @info """Best ind in $folder_to_read found in $(best.root).
            best loss : $(best.loss). Others losses (min, mean): $others_min, $others_mean
            """
        else
            @info """Best ind in $folder_to_read found in $(best.root).
            best loss : $(best.loss).
            """
        end
        # read ind
        ind_path = joinpath(best.root, "checkpoint_0.pickle")
        best_ind = deserialize(ind_path)["best_genome"]
        k = match(r"[0-9]_[0-9]*", best.root).match
        K = (parse(Int, k[1]), parse(Int, k[3:end]))
        Files_Losses[best.root] = best.loss
        Bests[K] = (loss = best.loss, path = ind_path, root = best.root, ind = best_ind)
    else
        # reading all
        best = best_losses[begin]
        @info """Best ind in $folder_to_read found in $(best.root).
        best loss : $(best.loss). Reading all.
        """
        for ind_info in best_losses
            ind_path = joinpath(ind_info.root, "checkpoint_0.pickle")
            best_ind = deserialize(ind_path)["best_genome"]
            k = match(r"[0-9]_[0-9]*", best.root).match
            K = (parse(Int, k[1]), parse(Int, k[3:end]))
            place = get!(Bests, K, [])
            push!(place, (loss = best.loss, path = ind_path, root = best.root, ind = best_ind))
        end
    end
end

# Extract Inds :
ks = keys(Bests) |> collect |> sort

if use_bests
    pop = [Bests[k].ind for k in ks]
else
    tmp = reduce(vcat, [Bests[k] for k in ks])
    pop = map(i -> i.ind, tmp)
    @info "Pop length : $(length(pop))"
end

N_NODES = pop[1][1] |> length
node_config = nodeConfig(N_NODES, 1, 3, n_ins)
shared_in, _ = make_evolvable_utgenome(
    model_arch, ml, node_config
)

# TRAIN DATA
nt = Threads.nthreads()

# Save best modules
UTCGP.reset_genome!.(pop)
folder_pop = joinpath(folder, "best_modules")
isdir(folder_pop) || mkdir(folder_pop)
open(joinpath(folder_pop, "best_modules_bests$(use_bests).pickle"), "w") do io
    write(io, UTCGP.general_serializer(pop))
end

for (datax, datay, split) in [(trainx, trainy, "train"), (valx, valy, "val")]
    OUTS_ALL = []
    for (i, ind) in enumerate(pop)
        @info "Running $i"
        local K = (1, i)
        best_ind = ind
        xs = datax
        ys = datay
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
                    out_v[sample_idx] = (pred = outputs[1][1], gt = y)
                end
            end
            push!(tasks, t)
        end
        fetch.(tasks)
        push!(OUTS_ALL, deepcopy(OUTS))
    end


    OUTS_PREDS = []
    for (i, pack) in enumerate(OUTS_ALL)
        push!(OUTS_PREDS, [j.pred for j in pack])
    end

    OUTS_GT = []
    for (i, pack) in enumerate(OUTS_ALL)
        push!(OUTS_GT, [j.gt for j in pack])
    end

    # Make dataset for training DT, RF etc
    mage_preds = reduce(hcat, OUTS_PREDS)
    gts = OUTS_GT[1]
    initial_x = map(z -> map(i -> reinterpret.(UInt8, i.img), z), datax)
    payload = (
        ys = [mage_preds[i, :] for i in 1:size(mage_preds)[1]],
        gt = gts,
        extras = (
            sruct_ = CLASSIFICATION_DATASET_VEC_SCALAR,
            initial_x = initial_x,
        ),
    )
    save_path = joinpath(folder_pop, "best_modules_dataset_$(split)_bests$(use_bests).jld2")
    @info "Saving dataset to $save_path"
    save_object(save_path, payload)
end
