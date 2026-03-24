using CUDA, cuDNN
using PartialFunctions
using PythonCall

st = time()
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))

include(joinpath(home, "src", "magenet_imports.jl"))
include(joinpath(home, "src", "magenet_args.jl"))
include(joinpath(home, "src", "magenet_ski.jl"))

@add_arg_table s begin
    "--boost_round"
    arg_type = Int
end
Parsed_args = parse_args(s)
@show Parsed_args

BOOST_ROUND = Parsed_args["boost_round"]
@info "BOOSTING ROUND $BOOST_ROUND"
const_vars = setup_constants(Parsed_args)
map(i -> eval(i), const_vars)

USE_SKI ? addprocs(nt, exeflags = ["--threads=1"]) : nothing
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images
# pinthreads(:cores) ; println(threadinfo(; slurm = true, color = false))

MAGENetwork.FREEZE_RATE[] = FREEZE_RATE
MAGENetwork.BS[] = BATCH_SIZE
MAGENetwork.λ_cosinesim[] = REGULARIZATION
MAGENetwork.DP_RATE[] = DROPOUT_RATE
MAGENetwork.RESET_PB[] = RESET_PB
MAGENetwork.OPTIM[] = OPTIM

# const _device = setup_cuda_device(DEVICE_CU)
include_utils_and_disable_logging(home)

Parsed_args["type_of_module"] = "v1"
type_of_module = Symbol(Parsed_args["type_of_module"])

# ACTIVATION
if Parsed_args["act"] == "asinh"
    MAGENetwork.MODULE_ACT[] = asinh
else
    throw(error("implement others"))
end

######################
# READ FILES #########
######################
data_path = Parsed_args["data_location"]
data = load(data_path)["single_stored_object"]
trainx, trainy, train_inter_features = data.extras.initial_x, data.gt, data.ys
trainx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in trainx]

val_data_path = Parsed_args["val_data_location"]
if val_data_path != ""
    data = load(val_data_path)["single_stored_object"]
    valx, valy, val_inter_features = data.extras.initial_x, data.gt, data.ys
    valx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in valx]
    @assert length(valx) == length(valy)
else
    valx, valy = nothing, nothing
end

@assert length(trainx) == length(trainy) == length(train_inter_features)
@assert length(valx) == length(valy) == length(val_inter_features)
@assert length(train_inter_features[1]) == length(val_inter_features[1])
@info "Size train : $(length(trainx))"
@info "Size val : $(length(valx))"
@info "Inter Features dimensions: $(length(train_inter_features[1]))"

const CLASSES = sort(unique(trainy))
const N_CLASSES = length(CLASSES)
const sample_img = trainx[1][1]

# READING ALL NN MODELS
NN_PREDS = Vector{Float64}[]
experiment_path = joinpath(Parsed_args["output_dir"], Parsed_args["trial_id"])
nn_paths = filter(x -> occursin("nn_", x), readdir(experiment_path))
for nn_folder in nn_paths
    nn_path = joinpath(experiment_path, nn_folder)
    @info "reading nn in $nn_path"
    for (path, dirs, files) in walkdir(nn_path)
        files_to_read = filter(x -> occursin("surrogate", x), files)
        files_to_read = filter(x -> occursin("train", x), files_to_read)
        for file in files_to_read
            file = joinpath(path, file)
            @info "Reading file $file"
            data = load(file)["single_stored_object"]
            push!(NN_PREDS, data.ys)
        end
    end
end

# Image Bundles
include(joinpath(home, "src", "magenet_image_bundles.jl"))
define_common_image_functions(sample_img)
skimage_factories = USE_SKI ? setup_skimage_distributed(Type2Dimg_binary) : nothing
N_INS = trainx[1] |> length

# Float Bundles
float_bundles = UTCGP.get_float_bundles()
USE_SKI ? push!(float_bundles, skimage_factories...) : nothing
push!(float_bundles, bundle_float_imagegraph) # TODO
glcm_b = deepcopy(UTCGP.experimental_bundle_float_glcm_factory) # TODO
push!(float_bundles, glcm_b)

# Symbolic Regression Bundles
only_float_bundles = UTCGP.get_sr_float_bundles()

set_bundle_casters!(float_bundles, float_caster2)
set_bundle_casters!(only_float_bundles, float_caster2)

# Metalibs
ml = ml_from_vbundles([image_intensity, image_binary, image_segment, float_bundles])
ml_float = ml_from_vbundles([only_float_bundles])

# PYTORCHLIP BACKEND
pylipbackend = MAGENetwork.get_pytorchlip_backend()
pylipbackend.torch.set_default_dtype(pylipbackend.torch.float32)

# Make new pop
initial_pop = create_initial_magenet_population(
    pylipbackend,
    N_ELITE,
    Parsed_args, Type2Dimg_intensity, (Type2Dimg_binary, Type2Dimg_segment), ml, ml_float, valx, N_NODES, N_CLASSES; type_of_module
)

rc = MNRunConf(
    ; gens = GENS, n_elite = N_ELITE, n_new = N_NEW, ts = TOUR_SIZE,
    mutation_n_models = MUTATION_N_MODELS,
    mutation_model = MUTATION_RATE
)
endpoint = PopVsSampleCLS{IndVsSample}(N_CLASSES)

# KEEP ONLY FIRST TREE CHANNELS
inputs_nn = trainx
labels_nn = trainy
program_data_nn = MAGENetwork.MAGEProgramInstance[]
for (x, y) in zip(inputs_nn, labels_nn)
    push!(program_data_nn, MAGENetwork.MAGEProgramInstance(x[1:3], y)) # keep only first 3 channels
end
@info "Number of training samples : $(length(program_data_nn))"

val_inputs_nn = valx
val_labels_nn = valy
val_program_data_nn = MAGENetwork.MAGEProgramInstance[]
for (x, y) in zip(val_inputs_nn, val_labels_nn)
    push!(val_program_data_nn, MAGENetwork.MAGEProgramInstance(x[1:3], y))
end
@info "Number of val samples : $(length(val_program_data_nn))"

# PYTORCH AND NUMPY
torch = pyimport("torch")
np = pyimport("numpy")
torchvision = pyimport("torchvision.transforms")
PIL = pyimport("PIL.Image")

cifar_augment = torchvision.AutoAugment(torchvision.autoaugment.AutoAugmentPolicy.CIFAR10)
py_train_transforms = torchvision.Compose(
    [
        torchvision.ToPILImage("RGB"), # julia array passed is already (HxWxC) in Float32 so is has to be converted to uint8 => pil
        cifar_augment,
        #torchvision.RandomResizedCrop(size(sample_img), scale = (0.8, 1.0), ratio = (0.8, 1.2)),
        #torchvision.RandomHorizontalFlip(),
        #torchvision.RandomRotation(10, expand = false),
        #torchvision.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.1),
        torchvision.ToTensor(),   # converts to Float32 [0,1]
        torchvision.Normalize(                   # normalize with CIFAR-10 mean/std
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.247, 0.2435, 0.2616]
        ),
    ]
)

py_val_transforms = torchvision.Compose(
    [
        torchvision.ToTensor(),   # converts to Float32 [0,1]
        torchvision.Normalize(                   # normalize with CIFAR-10 mean/std
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.247, 0.2435, 0.2616]
        ),

    ]
)

# JULIA-PYTORCH DATA COM
function prepare_for_julia(x::Py) # torch tensor
    nparray = x.numpy()
    jl_array = pyconvert(Array{Float32, 3}, nparray) # CxHxW
    # clamp01nan!(jl_array)
    return jl_array # CxHxW
end

# function split_to_channels(x::Array{Float32, 3})
# sizes = size(x)
# @assert sizes[1] < sizes[2] # channels first
# to change to RGB we have to make CxHxW
# return permutedims(x, (2, 3, 1)) # flux expects HxWxC
# end

function apply_transformations(transformations, x)
    # x is HxWxC, float 32
    nparray = np.asarray(x, copy = false) # in any case pil makes a copy
    transformed_rgb = transformations(nparray)
    return ready_for_julia = prepare_for_julia(transformed_rgb)
    # return split_to_channels(ready_for_julia) #HxWxC
end

train_transformations = apply_transformations $ py_train_transforms
val_transformations = apply_transformations $ py_val_transforms

m = initial_pop[1]

# HIJACK SECOND LAYER ---
n_modules_l2 = m[2] |> length
type_of_nn_needed = MAGENetwork._infer_nn_model_type_based_on_layer(m[2].ma)
for ith_module in 1:n_modules_l2
    original = m[2][ith_module].surrogate.model
    m[2][ith_module].surrogate.model = MAGENetwork.create_nn_model(
        pylipbackend, type_of_nn_needed,
        (length(train_inter_features[1]) + Parsed_args["l1_size"]), # features by mage + nn being trained
        original.output_dim; size = original.size, last = false #original.is_last
    )
end

# TRAIN
MAGENetwork.LR[] = 0.001
MAGENetwork.BS[] = 2048

outs = MAGENetwork.extend_feature_space(
    pylipbackend, m,
    program_data_nn, val_program_data_nn,
    train_inter_features, val_inter_features,
    reduce(hcat, NN_PREDS);
    n_classes = N_CLASSES,
    max_epochs = 25,
    transformations = train_transformations,
    val_transformations = val_transformations,
);

# indices = outs[2].indices
model = outs[1]

# # SAVE MODEL ---
n_classes = N_CLASSES
prepared_train_batches = MAGENetwork.prepare_nn_dataloader(program_data_nn, MAGENetwork.BS[], MAGENetwork.ImagesToScalarNN, n_classes; transformations = val_transformations, shuffle = false)
acc, _ = MAGENetwork.test_parallel_model(
    pylipbackend,
    model,
    prepared_train_batches,
    train_inter_features
)
@info acc

prepared_val_batches = MAGENetwork.prepare_nn_dataloader(val_program_data_nn, MAGENetwork.BS[], MAGENetwork.ImagesToScalarNN, n_classes; transformations = val_transformations, shuffle = false)
val_acc, _ = MAGENetwork.test_parallel_model(
    pylipbackend,
    model,
    prepared_val_batches,
    val_inter_features
)
@info val_acc

# CAPTURE TRAIN
captured_train_data = MAGENetwork.capture_surrogate_training_data(
    pylipbackend, model, prepared_train_batches, train_inter_features
)
captured_val_data = MAGENetwork.capture_surrogate_training_data(
    pylipbackend, model, prepared_val_batches, val_inter_features
)

# MAKE FOLDERS TO SAVE
folder = Parsed_args["output_dir"]
isdir(folder) || mkdir(folder)
id = Parsed_args["trial_id"]
folder = joinpath(folder, id)
isdir(folder) || mkdir(folder)
folder = joinpath(folder, "nn_surrogates_boost_$BOOST_ROUND")
isdir(folder) || mkdir(folder)

# SAVE ONE BY ONE
for (split, real_data, split_str) in ((captured_train_data, program_data_nn, "train"), (captured_val_data, val_program_data_nn, "val"))
    xs = map(z -> map(i -> reinterpret.(UInt8, i.img), z.inputs), real_data) # saving as utf8 uses way more space
    for i in 1:length(model[1]) # only save first layer
        k = (1, i)
        ys = map(x -> x.outputs, split[k])
        @show std(ys)
        @assert length(ys) == length(xs)
        tmp = (
            xs = xs,
            ys = ys,
            extras = (
                :struct_ => SurrogateDataset_IMG_SCALAR,
            ),
        )
        p = joinpath(folder, "surrogate_$(k[1])_$(k[2])_$(split_str).jld2")
        save_object(p, tmp)
        @info "Data saved for $k for $split_str : $(length(xs))"
    end
end

open(joinpath(folder, "metrics.txt"), "w") do f
    write(f, "$acc, $val_acc\n")
end

@pyeval (w = outs[2].best_weights, filename = joinpath(folder, "model_weights.pt")) => "torch.save(w, filename)"

# LOAD WEIGHTS WITH
# @pyeval (
#     m = chain_surrogate,
#     torch = pylipbackend.torch, filename = "lip_cifar_exp/EXP_1/nn_surrogates/model_weights.pt", loc = "cpu",
# ) => "m.load_state_dict(torch.load(filename, map_location = loc))"
