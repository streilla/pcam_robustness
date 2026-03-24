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
Parsed_args = parse_args(s)
@show Parsed_args
include(joinpath(home, "src", "magenet_ski.jl"))

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
MAGENetwork.MODULE_ACT[] = sigmoid

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

MAGENetwork.LR[] = 0.0025
MAGENetwork.BS[] = 2048

outs = MAGENetwork.train_model_alone(
    pylipbackend, m,
    program_data_nn, val_program_data_nn;
    n_classes = N_CLASSES,
    max_epochs = 20,
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
    prepared_train_batches
)
@info acc
prepared_val_batches = MAGENetwork.prepare_nn_dataloader(val_program_data_nn, MAGENetwork.BS[], MAGENetwork.ImagesToScalarNN, n_classes; transformations = val_transformations, shuffle = false)
val_acc, _ = MAGENetwork.test_parallel_model(
    pylipbackend,
    model,
    prepared_val_batches
)
@info val_acc

# CAPTURE TRAIN
captured_train_data = MAGENetwork.capture_surrogate_training_data(
    pylipbackend, model, prepared_train_batches
)
captured_val_data = MAGENetwork.capture_surrogate_training_data(
    pylipbackend, model, prepared_val_batches
)

# MAKE FOLDERS TO SAVE
folder = Parsed_args["output_dir"]
isdir(folder) || mkdir(folder)
id = Parsed_args["trial_id"]
folder = joinpath(folder, id)
isdir(folder) || mkdir(folder)
folder = joinpath(folder, "nn_surrogates")
isdir(folder) || mkdir(folder)

# SAVE ONE BY ONE
for (split, real_data, split_str) in ((captured_train_data, program_data_nn, "train"), (captured_val_data, val_program_data_nn, "val"))
    xs = map(z -> map(i -> reinterpret.(UInt8, i.img), z.inputs), real_data) # saving as utf8 uses way more space
    for i in 1:length(model[1]) # only save first layer
        k = (1, i)
        ys = map(x -> x.outputs, split[k])
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

# FROM CHANNELS TO COLOR VIEW

# function to_rgb(x) # for vector of simageND
#     xs = [reinterpret(z.img) for z in x]
#     return colorview(RGB, permutedims(cat(xs..., dims = 3), (3, 1, 2)))
# end

# function to_rgb(xs...) # for vector of uint8
#     xs = [reinterpret.(N0f8, x) for x in xs]
#     return colorview(RGB, permutedims(cat(xs..., dims = 3), (3, 1, 2)))
# end

# RESNET
# using Metalhead
# # using Infiltrator
# using PartialFunctions
# resnet = Metalhead.ResNet(18; nclasses = N_CLASSES, inchannels = N_INS)
# resnet18_dropout = Metalhead.resnet(
#     Metalhead.basicblock,
#     [2, 2, 2, 2];
#     inchannels = N_INS,
#     nclasses = N_CLASSES,
#     dropout_prob = 0.1,
#     norm_layer = Flux.InstanceNorm,
#     imsize = size(sample_img)
# )
# function resize_to_size(x, img_idx, args...)
#     new_sized_img = Images.imresize(float(x), (224, 224))
#     return SImageND(IntensityPixel{Float32}.(new_sized_img))
# end
# transformations = Function[mock_normalize]

# stem = Chain(
#     Conv((3, 3), N_INS => 64, stride = 1, pad = 1),
#     BatchNorm(64, relu)
# )
# get_layers = Metalhead.basicblock_builder(
#     [2, 2, 2, 2]; inplanes = 64, reduction_factor = 1,
#     activation = relu,
#     norm_layer = BatchNorm, revnorm = false, attn_fn = planes -> identity,
#     dropblock_prob = nothing, stochastic_depth_prob = nothing,
#     stride_fn = Metalhead.resnet_stride,
#     planes_fn = Metalhead.resnet_planes,
#     downsample_tuple = (Metalhead.downsample_conv, Metalhead.downsample_identity)
# )
# classifier_fn = nfeatures -> Metalhead.Layers.create_classifier(
#     nfeatures, N_CLASSES; dropout_prob = 0.5,
#     pool_layer = AdaptiveMeanPool((1, 1)), use_conv = false
# )
# custom_resnet = Metalhead.build_resnet(
#     (size(sample_img)..., N_INS), stem, get_layers, [2, 2, 2, 2],
#     Metalhead.addact $ relu, classifier_fn
# )

# image_means = Float32[]
# image_stds = Float32[]
# for c in 1:N_INS
#     s1, s2 = size(sample_img)
#     all_imgs = []
#     @info "Reading channel $c"
#     for img_label in program_data_nn
#         image = img_label.inputs[c].img |> float
#         push!(all_imgs, image)
#     end
#     @info "Done reading channel $c"
#     tmp = Array{Float32}(undef, s1, s2, length(all_imgs))
#     for (i, content) in enumerate(all_imgs)
#         tmp[:, :, i] .= content
#     end
#     @info size(tmp)
#     μ = mean(tmp)
#     σ = std(tmp)
#     push!(image_means, μ)
#     push!(image_stds, σ)
# end


# WARMUP ENZYME
# model = resnet18_dropout |> gpu
# dup_model = Enzyme.Duplicated(model)
# random_x = rand(Float32, 32, 32, 3, 1) |> gpu
# Flux.gradient(dup_model, Enzyme.Const(random_x)) do m, x
#     sum(m(x))
# end

# model = custom_resnet |> gpu
# random_x = rand(Float32, 32, 32, 3, 1) |> gpu
# Flux.gradient(model, random_x) do m, x
#     sum(m(x))
# end


# MAGENetwork.LR[] = 0.0001
# MAGENetwork.train_resnet_baseline(
#     program_data_nn, val_program_data_nn, custom_resnet;
#     n_classes = N_CLASSES,
#     batch_size = 256,
#     transformations = train_transformations,
#     val_transformations = val_transformations,
#     max_epochs = 100,
#     enzyme = false
# )


# struct DataLoader4{T<:Union{MLUtils.ObsView,MLUtils.BatchView},B,P,C,O,R<:AbstractRNG}
#     data::O  # original data
#     _data::T # data wrapped in ObsView / BatchView
#     batchsize::Int
#     buffer::B    # boolean, or external buffer
#     partial::Bool
#     shuffle::Bool
#     parallel::Bool
#     collate::C
#     rng::R
# end

# function Base.iterate(d::DataLoader4{T,Bool,:serial}) where {T}
#     @assert d.buffer == false
#     data = d.shuffle ? _shuffledata(d.rng, d._data) : d._data
#     iter = (getobs(data, i) for i in 1:numobs(data))
#     obs, state = iterate(iter)
#     return obs, (iter, state)
# end

# function Base.iterate(::DataLoader4, (iter, state))
#     ret = iterate(iter, state)
#     isnothing(ret) && return
#     obs, state = ret
#     return obs, (iter, state)
# end

# Base.length(d::DataLoader4) = numobs(d._data)
# Base.size(d::DataLoader4) = (length(d),)
# Base.IteratorEltype(d::DataLoader4) = Base.EltypeUnknown()

# function DL(data)
# collate = Val(nothing)
# buffer = false
# partial = false
# parallel = false
# batchsize = 100
# shuffle = false
# rng::AbstractRNG = Random.default_rng()
# _data = ObsView(data, collect(1:numobs(data)))
# _data = BatchView(_data; batchsize , partial, collate)
# P = :serial
# T, O, B, C, R = typeof(_data), typeof(data), typeof(buffer), typeof(collate), typeof(rng)
# @show T, O, B, C, R

# # MLUtils.DataLoader(data, _data, batchsize, buffer,
#                                 # partial, shuffle, parallel, collate, rng)
# DataLoader4{T,B,P,C,O,R}(data, _data, batchsize, buffer,
#                                 partial, shuffle, parallel, collate, rng)

# end
#
#

# using CUDA
# using Flux
# using Enzyme

# model = Chain(Dense(28^2 => 32, sigmoid), Dense(32 => 10), softmax);
# dup_model = Duplicated(model |> gpu);

# x1 = randn32(28 * 28, 1) |> gpu;
# y1 = [i == 3 for i in 0:9] |> gpu;
# grads_f = Flux.gradient((m, x, y) -> sum(abs2, m(x) .- y), dup_model, Const(x1), Const(y1))


# # REACTANT example


# using cuDNN
# using Reactant
# using Reactant_jll
# using MLDataDevices
# using CUDA
# CUDA.versioninfo()
# using Flux
# using Enzyme
# # const gpu_dev = MLDataDevices.reactant_device()
# # const gpu_dev = MLDataDevices.gpu_device(1)
# gpu_dev = gpu

# model = Chain(Dense(28^2 => 32, sigmoid), Dense(32 => 10), softmax);
# x1 = randn32(28 * 28, 1) |> gpu_dev;
# y1 = [i == 3 for i in 0:9] |> gpu_dev;

# dup_model = Duplicated(model)
# dup_model = dup_model |> gpu_dev

# model_compiled = @compile dup_model(x1)


# using cuDNN
# using Reactant
# using MLDataDevices
# using CUDA
# using Flux
# using Enzyme
# const gpu_dev = MLDataDevices.reactant_device()

# model = Chain(Dense(28^2 => 32, sigmoid), Dense(32 => 10), softmax);
# x1 = randn32(28 * 28, 1) |> gpu_dev;
# y1 = [i == 3 for i in 0:9] |> gpu_dev;

# dup_model = Duplicated(model)
# dup_model = dup_model |> gpu_dev

# f = (m, x, y) -> sum(abs2, m(x) .- y)
# function calc_grad(dup_model, x, y, f)
#     return Enzyme.gradient(
#         ReverseWithPrimal, f, dup_model, Const(x), Const(y)
#     )
# end
# gradient_compiled = @compile calc_grad(dup_model, x1, y1, f)
# res = gradient_compiled(dup_model, x1, y1, f)
# res.derivs[1].dval
# res.derivs[1].val
