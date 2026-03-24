st = time()
#ENV["JULIA_DEBUG"] = "UTCGP"
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))

include(joinpath(home, "src", "magenet_imports.jl"))
include(joinpath(home, "src", "magenet_args.jl"))
include(joinpath(home, "src", "magenet_ski.jl"))

const_vars = setup_constants(Parsed_args)
map(i -> eval(i), const_vars)

USE_SKI ? addprocs(nt, exeflags = ["--threads=1"]) : nothing
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images
# pinthreads(:cores) ; println(threadinfo(; slurm = true, color = false))

# setup_magenetwork_constants(
#     TH_FOR_ALIGN, TH_FOR_ES, FREEZE_RATE, BATCH_SIZE, REGULARIZATION, INTER_LOSSES, DROPOUT_RATE, RESET_PB, OPTIM, ACT, MAX
# )

const _device = setup_cuda_device(DEVICE_CU)
include_utils_and_disable_logging(home)

Parsed_args["n_elite"] = 1; @info "Setting n_elite to 1 since not needed"

######################
# READ FILES #########
######################
val_location = Parsed_args["val_data_location"]
has_val_data = val_location != ""

if has_val_data
    path_to_train_data = Parsed_args["data_location"]
    train_data = load(path_to_train_data)["single_stored_object"]
    val_data = load(val_location)["single_stored_object"]
    (trainx, trainy) = train_data.xs, train_data.ys
    (valx, valy) = val_data.xs, val_data.ys
else
    path_to_data = Parsed_args["data_location"]
    all_train_data = load(path_to_data)["single_stored_object"]
    (trainx, trainy), (valx, valy) = MLUtils.splitobs((all_train_data.xs, all_train_data.ys); at = 0.8, shuffle = true, stratified = all_train_data.ys)
end

const TRAINDataloader = make_dataloader(trainx, trainy, NSAMPLES, nt)
const VALDataloader = make_dataloader(valx, valy, NSAMPLES, nt)

const N_CLASSES = length(unique(valy))
const sample_img = trainx[1][1]
define_common_image_functions(sample_img)
skimage_factories = USE_SKI ? setup_skimage_distributed(nt, Type2Dimg) : nothing

# Image Bundles
include(joinpath(home, "src", "magenet_image_bundles.jl"))

# Float Bundles
float_bundles = UTCGP.get_float_bundles()
USE_SKI ? push!(float_bundles, skimage_factories...) : nothing
# push!(float_bundles, bundle_float_imagegraph) # TODO
glcm_b = deepcopy(UTCGP.experimental_bundle_float_glcm_factory) # TODO
push!(float_bundles, glcm_b)

# Symbolic Regression Bundles
only_float_bundles = UTCGP.get_sr_float_bundles()

set_bundle_casters!(float_bundles, float_caster2)
set_bundle_casters!(only_float_bundles, float_caster2)

# Metalibs
ml = ml_from_vbundles([image_intensity, image_binary, image_segment, float_bundles])
ml_float = ml_from_vbundles([only_float_bundles])

# Make new pop
initial_pop = create_initial_magenet_population(N_ELITE, Parsed_args, Type2Dimg_intensity, (Type2Dimg_binary, Type2Dimg_segment), ml, ml_float, valx, N_NODES, N_CLASSES)

rc = MNRunConf(
    ; gens = GENS, n_elite = N_ELITE, n_new = N_NEW, ts = TOUR_SIZE,
    mutation_n_models = MUTATION_N_MODELS,
    mutation_model = MUTATION_RATE
)
endpoint = PopVsSample(N_CLASSES)

# TRAIN MODEL
inputs_nn = trainx
labels_nn = trainy
program_data_nn = MAGENetwork.MAGEProgramInstance[]
for (x, y) in zip(inputs_nn, labels_nn)
    push!(program_data_nn, MAGENetwork.MAGEProgramInstance(x, y))
end
@info "Number of training samples : $(length(program_data_nn))"

val_inputs_nn = valx
val_labels_nn = valy
val_program_data_nn = MAGENetwork.MAGEProgramInstance[]
for (x, y) in zip(val_inputs_nn, val_labels_nn)
    push!(val_program_data_nn, MAGENetwork.MAGEProgramInstance(x, y))
end
@info "Number of val samples : $(length(val_program_data_nn))"

# Transformations
function mock_normalize(x, img_idx, args...)
    return SImageND(IntensityPixel{Float32}.(x))
end
transformations = Function[mock_normalize]

m = initial_pop[1]
mm = MAGENetwork.build_surrogate_model(m)
weights = Flux.trainables(mm)
MAGENetwork.normalize_all_weights!(weights)

MAGENetwork.with_device(
    model -> begin
        MAGENetwork.train_model_alone(
            m, program_data_nn, val_program_data_nn;
            n_classes = N_CLASSES,
            max_epochs = 1,
            transformations = transformations
        )
    end, m, :gpu
)

outs = MAGENetwork.with_device(
    model -> begin
        MAGENetwork.train_model_alone(
            model, program_data_nn, val_program_data_nn;
            n_classes = N_CLASSES,
            max_epochs = 8,
            transformations = transformations
        )
    end, m, :gpu
)

indices = outs[1].indices


# FIT ONCE

# program_data_nn
# val_program_data_nn
# elite = int
# elite_population = [elite]
# train_align_data = [MAGENetwork.collect_training_data(elite, inputs_nn) for elite in elite_population]
# val_align_data = [MAGENetwork.collect_training_data(elite, val_inputs_nn) for elite in elite_population]

# next_gen = MAGENetwork.strategy_three(
#     elite_population,
#     program_data_nn, val_program_data_nn, # GT instances
#     train_align_data, val_align_data, # Intermediary Outputs for alignment
#     val_program_data_nn;
#     k = 2,
#     device = :gpu,
#     epochs = 100,
#     trainsize_mage = 10_000,
#     lambda_mage = 20,
#     n_classes = 10,
#     transformations = transformations
# )

# IMPROVE AND CATCH LOOP --- --- ---

m = initial_pop[1]

reset_random_weights_model!(m)

MAGENetwork.TH[] = 0.0000000000001
MAGENetwork.FREEZE_RATE[] = 0.0
MAGENetwork.OPTIM[] = "adam"
MAGENetwork.DP_RATE[] = 0.1
MAGENetwork.ACT[] = false
MAGENetwork.λ[] = 0.0001f0
MAGENetwork.INTER_LOSSES[] = 0.5f0
MAGENetwork.LR[] = 0.001

elite_population = [m]
n_classes = 10
improve_data = val_program_data_nn # an argument for train data is also to be made
trainsize_mage = 40
model_registry = MAGENetwork.model_registry
model_uuids = MAGENetwork.model_uuids
parent_map = MAGENetwork.parent_map
device = :gpu
epochs = 5000
lambda_mage = 10

n_batches_train = 40
n_batches_val = 70

train_nn = true
train_last_layer = true
device_ = :gpu

for ith_iteration in 1:30
    TH, STORAGE = MAGENetwork.TH, MAGENetwork.STORAGE
    M = STORAGE[]
    if !haskey(STORAGE[], "mage_alignment") # which NNs (row,col) were free
        M["mage_alignment"] = []
    end

    # Sample data for training / val nns
    # examples = TRAINDataloader[1:(MAGENetwork.BS[] * n_batches_train)]
    # @info length(examples)
    # inputs_nn = [x[1] for x in examples]
    # labels_nn = [x[2][1] for x in examples]
    # program_data_nn = MAGENetwork.MAGEProgramInstance[]
    # MAGEProgramInstance[]

    # for (x, y) in zip(inputs_nn, labels_nn)
    #     push!(program_data_nn, MAGENetwork.MAGEProgramInstance(x, y))
    # end

    # val_examples = VALDataloader[1:(MAGENetwork.BS[] * n_batches_val)]
    # @info length(val_examples)
    # val_inputs_nn = [x[1] for x in val_examples]
    # val_labels_nn = [x[2][1] for x in val_examples]
    # val_program_data_nn = MAGENetwork.MAGEProgramInstance[]
    # for (x, y) in zip(val_inputs_nn, val_labels_nn)
    #     push!(val_program_data_nn, MAGENetwork.MAGEProgramInstance(x, y))
    # end
    train_align_data = [MAGENetwork.collect_training_data(elite, inputs_nn) for elite in elite_population]
    val_align_data = [MAGENetwork.collect_training_data(elite, val_inputs_nn) for elite in elite_population]

    # Sample data for capturing improvement behavior
    idx_subset_of_examples = sample(1:length(improve_data), trainsize_mage)
    subset_of_examples = improve_data[idx_subset_of_examples]
    gt_inputs_of_examples = [i.inputs for i in subset_of_examples]
    gt_outputs_of_examples = [i.outputs for i in subset_of_examples]

    all_gt_inputs = [i.inputs for i in improve_data]
    all_gt_outputs = [i.outputs for i in improve_data]

    # Starting population (clones of original elites)
    all_variants = Vector{MNModel}()

    empty!(model_registry)
    empty!(model_uuids)
    empty!(parent_map)

    old_model = elite_population[1]
    model = deepcopy(old_model) # also copy weights from old to new
    MAGENetwork.turn_on!(model)

    active_surrogates = []
    weights = Float64[]
    for (layer_idx, layer) in enumerate(model.mnsequence.mnlayers)
        for (prog_idx, mage_and_surrogate) in enumerate(layer.programs)
            if mage_and_surrogate.surrogate isa MAGENetwork.NNSurrogateModel &&
                    mage_and_surrogate.surrogate.active
                push!(active_surrogates, (layer_idx, prog_idx))
                push!(weights, mage_and_surrogate.surrogate.loss)
            end
        end
    end

    # if ith_iteration > 1 && train_nn
    if train_nn
        if ith_iteration == 1
            MAGENetwork.INTER_LOSSES[] = 0.001f0
        else
            MAGENetwork.INTER_LOSSES[] = 0.01f0
        end
        MAGENetwork.FREEZE_RATE[] = 1.0
        # MAGENetwork.with_device(
        #     model -> begin
        #         MAGENetwork.train_model_alone(
        #             m, program_data_nn, val_program_data_nn;
        #             n_classes = N_CLASSES,
        #             max_epochs = 2,
        #             transformations = transformations
        #         )
        #     end, m, :gpu
        # )
        model, weight_changes, align_after = MAGENetwork.with_device(
            model -> begin
                MAGENetwork.joint_training!(
                    model,
                    train_align_data, val_align_data,
                    program_data_nn, val_program_data_nn,
                    device = device_,
                    k_to_train = nothing,
                    n_classes = n_classes,
                    transformations = transformations
                )
            end, model, device_
        )
    end


    normalize_or_not = isnothing(transformations) ? nothing : transformations[1:1]
    surrogate_data = MAGENetwork.with_device(
        model -> begin
            MAGENetwork.capture_surrogate_training_data(
                model, subset_of_examples, normalize_or_not
            )
        end,
        model,
        :gpu
    )
    orig_surrogate_data = deepcopy(surrogate_data)

    # KEYS of layer-prog
    ks = []
    for (layer_idx, layer) in enumerate(model.mnsequence.mnlayers)
        for (prog_idx, mage_and_surrogate) in enumerate(layer.programs)
            push!(ks, (layer_idx, prog_idx))
        end
    end

    # SHOW MAGENET DASHBOARD
    # # replace inputs from surrogate data to only capture outputs
    initial_inputs_for_magenet = gt_inputs_of_examples # NOTE is as surrogate[(1,.)] but that input might be transformed, so we picked the true input
    mage_capture = MAGENetwork.collect_training_data(model, initial_inputs_for_magenet, true)

    for k in ks
        layer_idx, prog_idx = k[1], k[2]
        captured_data = deepcopy(surrogate_data[k])
        mage_capture_layer_instances = MAGENetwork.prepare_program_training_data(mage_capture.layers[layer_idx], prog_idx, nothing, nothing) # this will give us the correct inputs
        for (instance_idx, instance) in enumerate(captured_data)
            new_inputs = deepcopy(mage_capture_layer_instances[instance_idx].inputs)
            empty!(instance.inputs)
            push!(instance.inputs, new_inputs...) # replace captured inputs with the true MAGENet inputs
            # NOTE outputs are what we will try to fit to
        end
        surrogate_data[k] = captured_data
    end

    # # Calc loss for every model
    vs = OrderedDict{Tuple{Int, Int}, Tuple{Float64, Float64}}()
    for (layer_idx, layer) in enumerate(model.mnsequence.mnlayers)
        for (prog_idx, mage_and_surrogate) in enumerate(layer.programs)
            layer_prog_key = (layer_idx, prog_idx)
            loss_func = MAGENetwork.get_mage_loss_function(layer.ma)
            p = mage_and_surrogate.program
            initial_fitness, losses = MAGENetwork.evaluate_mage_fitness(p, surrogate_data[layer_prog_key], layer.ma, layer.ml, layer.programs_input[], loss_func)
            outs = [surrogate_data[layer_prog_key][i].outputs for i in 1:trainsize_mage]
            if initial_fitness <= MAGENetwork.TH[] || std(outs) < 0.00001
                if isdefined(Main, :Infiltrator)
                    Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
                end
            end
            vs[layer_prog_key] = (initial_fitness, std(outs))
        end
    end
    MAGENetwork.print_mage_dashboard(vs)
    # push!(M["mage_alignment"], vs)

    selected_surrogates_train = collect(keys(filter(kv -> kv.second[1] > TH[], vs)))
    @info selected_surrogates_train
    # if isempty(selected_surrogates_train)
    #     @info "No Misalign higher than $TH"
    #     selected_surrogates_train = selected_surrogates
    # end

    # ALIGN ALL OUTPUTS
    @info "TRAINING ALL LAYERS TO ALIGN ON OUTPUT, ONE BY ONE"
    ks = collect(keys(vs))
    grouped_per_layer = OrderedDict{Int, Vector{Tuple{Int, Int}}}()
    for tup in ks
        key = tup[1]
        push!(get!(grouped_per_layer, key, []), tup)
    end
    max_k = maximum(collect(keys(grouped_per_layer)))

    CAPTURE = []
    for (layer, keys_to_train) in grouped_per_layer
        MAGENetwork._reset_mnmodel!(model)
        MAGENetwork.decode_mn!(model)

        layer_tasks = []
        n_to_train_in_this_layer = length(keys_to_train)
        mage_capture = MAGENetwork.collect_training_data(model, initial_inputs_for_magenet, true)
        threads = Threads.nthreads()
        tsize = Base.ceil(Int, n_to_train_in_this_layer / threads)

        push!(CAPTURE, deepcopy(mage_capture))

        # Partition the selected surrogates into chunks
        for idx in Iterators.partition(1:n_to_train_in_this_layer, tsize)
            t = @spawn begin
                for id in idx
                    tid = Threads.threadid()
                    layer_prog = keys_to_train[id]
                    layer_idx = layer_prog[1]
                    prog_idx = layer_prog[2]
                    init_loss = vs[layer_prog]
                    model_layer = model[layer_idx]
                    @assert layer == layer_idx

                    nn_to_nn_data = deepcopy(surrogate_data[layer_prog]) # we want the outputs from here
                    mage_capture_layer_instances = MAGENetwork.prepare_program_training_data(mage_capture.layers[layer], prog_idx, nothing, nothing) # this will give us the correct inputs
                    for (instance_idx, instance) in enumerate(nn_to_nn_data)
                        new_inputs = deepcopy(mage_capture_layer_instances[instance_idx].inputs)
                        empty!(instance.inputs)
                        push!(instance.inputs, new_inputs...) # replace captured inputs with the true MAGENet inputs
                    end
                    # TODO check difference between nn_to_nn_data and surrogate_data[layer_prog]. If layer == 1, should be the same. If not, inputs are not the same, as one
                    # has inputs from nns and the other from Magenet.
                    # if isdefined(Main, :Infiltrator)
                    #     Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
                    # end

                    initial_fitness, _ = MAGENetwork.evaluate_mage_fitness(
                        model_layer[prog_idx].program,
                        nn_to_nn_data, model_layer.ma, model_layer.ml, model_layer.programs_input[],
                        MAGENetwork.get_mage_loss_function(model_layer.ma)
                    )

                    @info "Loss in dash : $(init_loss). Maybe will train $layer_prog in $tid"
                    @info "Initial F : $layer_prog : $initial_fitness"
                    if initial_fitness < MAGENetwork.TH[] && layer_idx != max_k
                        @info "Skipping this prog $layer_prog"
                        continue
                    end

                    if layer_idx == max_k
                        # @info "Using ideal data for $layer_prog"
                        # # more_data_mage_capture = collect_training_data(model, all_gt_inputs, true)
                        # # more_mage_capture_layer_instances = prepare_program_training_data(more_data_mage_capture.layers[layer], prog_idx, nothing, nothing) # this will give us the correct inputs
                        # data_to_improve_prog = MAGEProgramInstance[]
                        # # also mock the outputs
                        # ideal_outputs_for_prog = Float64.(gt_outputs_of_examples .== prog_idx) #
                        # @assert length(mage_capture_layer_instances) == length(ideal_outputs_for_prog)
                        # e = epochs
                        # @info "Nb of samples for training last layer $(length(mage_capture_layer_instances)) for $e"
                        # for (instance_idx, instance) in enumerate(mage_capture_layer_instances)
                        #     xy = MAGEProgramInstance(
                        #         instance.inputs, # mage inputs # what the prog will see
                        #         ideal_outputs_for_prog[instance_idx] # ideal outputs that yield 100% acc
                        #     )
                        #     push!(data_to_improve_prog, xy)
                        # end
                        @info "Skipping Training for $layer_prog"
                    else
                        data_to_improve_prog = nn_to_nn_data
                        e = epochs
                        MAGENetwork.train_mage_from_surrogate_data!(model, layer_idx, prog_idx, data_to_improve_prog, e, lambda_mage)
                    end
                end
            end
            push!(layer_tasks, t)
        end
        fetch.(layer_tasks)
    end

    MAGENetwork.decode_mn!(model)
    vs = OrderedDict{Tuple{Int, Int}, Tuple{Float64, Float64}}()
    for (layer_idx, layer) in enumerate(model.mnsequence.mnlayers)
        for (prog_idx, mage_and_surrogate) in enumerate(layer.programs)
            layer_prog_key = (layer_idx, prog_idx)
            loss_func = MAGENetwork.get_mage_loss_function(layer.ma)
            p = mage_and_surrogate.program
            initial_fitness, _ = MAGENetwork.evaluate_mage_fitness(p, surrogate_data[layer_prog_key], layer.ma, layer.ml, layer.programs_input[], loss_func)
            n = length(surrogate_data[layer_prog_key])
            outs = [surrogate_data[layer_prog_key][i].outputs for i in 1:n]
            if initial_fitness <= MAGENetwork.TH[] || std(outs) < 0.00001
                if isdefined(Main, :Infiltrator)
                    Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
                end
            end
            vs[layer_prog_key] = (initial_fitness, std(outs))
        end
    end
    MAGENetwork.print_mage_dashboard(vs)


    # --- --- train last layer jointly
    programs_keys = grouped_per_layer[max_k]
    ideal_outputs_for_prog = gt_outputs_of_examples # NOTE this are the true labels for these input examples
    # inputs for the layer
    more_data_mage_capture = MAGENetwork.collect_training_data(model, initial_inputs_for_magenet, true)
    more_mage_capture_layer_instances = MAGENetwork.prepare_program_training_data(more_data_mage_capture.layers[max_k], 1, nothing, nothing) # NOTE we want to capture the inputs for (max_k, 1) since those inputs are common for all (max_k, i)

    # TODO collect training data still layer_cast all layers WARNING
    # if isdefined(Main, :Infiltrator)
    #     Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
    # end

    data = MAGEProgramInstance[]
    for (instance_idx, instance) in enumerate(more_mage_capture_layer_instances)
        xy = MAGEProgramInstance(
            instance.inputs, # mage inputs # what the prog will see
            ideal_outputs_for_prog[instance_idx] # true class
        )
        push!(data, xy)
    end

    MAGENetwork.decode_mn!(model)

    # copy improved weights back to old model
    # since there most be at least some amount of alignment, we want that alignment to be cummulative
    # copy_surrogate_weights!(old_model, model)

    # set_parent!(old_model, old_model)
    # set_parent!(model, model)
    # new_pop = [old_model, model]
    # @info "Out from Strat : $(get_model_uuid.(new_pop)). (OLD, NEW)"
    # @info "Out from Strat $(objectid.(new_pop))"

    # @info length(program_data_nn)
    # P = []
    # for i in program_data_nn
    #     outs = MAGENetwork.forward(model, i.inputs)
    #     push!(P, outs)
    # end
    # new_model_pred_cat = map(i -> argmax(i), P)
    # gt = [i.outputs for i in program_data_nn]
    # acc = mean(new_model_pred_cat .== gt)
    # @info "NEW MODEL ACC : $acc"

    # decode_mn!(old_model)
    # P_old = []
    # for i in program_data_nn
    #     outs = MAGENetwork.forward(old_model, i.inputs)
    #     push!(P_old, outs)
    # end
    # gt_model_pred_cat = map(i -> argmax(i), P_old)
    # gt = [i.outputs for i in program_data_nn]
    # old_acc = mean(gt_model_pred_cat .== gt)
    # @info "OLD MODEL ACC : $old_acc"

    elite_population[1] = model

    # TRAIN HEAD
    batch_size = MAGENetwork.BS[]
    updated_train_data = MAGENetwork.collect_training_data(model, map(i -> i.inputs, program_data_nn), true) # this already reflects the updated previous layers
    updated_val_data = MAGENetwork.collect_training_data(model, map(i -> i.inputs, val_program_data_nn), true) # this already reflects the updated previous layers
    network_structure_train, network_dataloader_train, network_prog_types = MAGENetwork.to_network_structure(updated_train_data, model, batch_size, device_, nothing)
    network_structure_val, network_dataloader_val, _ = MAGENetwork.to_network_structure(updated_val_data, model, batch_size, device_, nothing)

    ks = keys(network_structure_train)
    network_batched_train = OrderedDict()
    network_batched_val = OrderedDict()
    for k in ks
        network_batched_train[k] = [i for i in network_dataloader_train[k]]
        network_batched_val[k] = [i for i in network_dataloader_val[k]]
    end

    prepared_train_batches = MAGENetwork.prepare_nn_dataloader(
        program_data_nn, batch_size, MAGENetwork.ImagesToScalarNN, device_, n_classes,
        onehot = true, shuffle = false, transformations = nothing
    ) # TO NN format (FIRST INPUT - GROUND TRUTH)
    prepared_val_batches = MAGENetwork.prepare_nn_dataloader(
        val_program_data_nn, batch_size, MAGENetwork.ImagesToScalarNN, device_, n_classes;
        onehot = true, shuffle = false, transformations = nothing
    ) # TO NN format (FIRST INPUT - GROUND TRUTH)
    @show length(prepared_train_batches) length(prepared_val_batches)

    prev_layer = max_k - 1 # idx prev layer, before head
    n_outs_from_prev_layer = model[prev_layer].programs |> length
    prev_layer_keys = [(prev_layer, i) for i in 1:n_outs_from_prev_layer]
    n_outs = model[max_k].programs |> length
    heads = []
    for i in 1:n_outs
        nn = Flux.Chain(
            # LayerNorm(n_outs_from_prev_layer),
            Dense(n_outs_from_prev_layer => 4, relu),
            Dropout(0.1),
            Dense(4 => 3, relu),
            Dropout(0.1),
            Dense(3 => 2, relu),
            Dropout(0.1),
            Dense(2 => 1)
        )
        push!(heads, nn)
    end
    head_nn = Flux.Parallel(vcat, heads...)
    Flux.trainmode!(head_nn)

    epoch = 0
    budget = 10
    head_nn = head_nn |> gpu
    optimizer = AdamW(0.001f0)
    opt_state = Flux.setup(optimizer, head_nn)
    total_loss, train_losses, best_loss = 0.0, Float64[], Inf
    epochs_without_improvement = 0
    train_accuracies, val_accuracies = Float64[], Float64[]
    best_val = -Inf
    best_weights = deepcopy(Flux.state(head_nn))
    best_at = 0
    while epoch < 80 #&& epoch < budget
        epoch += 1
        epoch_loss, batch_count, epoch_acc = 0.0, 0, 0.0
        all_inter_losses_ = 0.0
        for (batch_idx, real_data) in enumerate(prepared_train_batches)
            prev_layer_mage_outs = vcat([network_batched_train[prev_k][batch_idx][2] for prev_k in prev_layer_keys]...) |> gpu
            parallel_inputs = ntuple(_ -> prev_layer_mage_outs, n_outs)
            real_labels = real_data[2] |> gpu
            (loss, losses_align, outputs), grads = Flux.withgradient(head_nn) do m
                outputs = m(parallel_inputs)
                cls_loss = MAGENetwork.classification_loss(outputs, real_labels)
                #reg = snorm(Flux.trainables(m))
                #scaled_reg = λ[] * reg
                final_loss = cls_loss # scaled_reg
                final_loss, (final_loss, cls_loss), outputs
            end
            epoch_loss += loss
            batch_count += 1
            opt_state, parallel_surrogate = Flux.Optimisers.update!(opt_state, head_nn, grads[1])

            # ACC
            preds = Flux.onecold(cpu(outputs))
            true_labels = Flux.onecold(cpu(real_labels))
            correct = sum(preds .== true_labels)
            epoch_acc += correct / length(true_labels)
        end

        avg_epoch_loss = epoch_loss / batch_count
        push!(train_losses, avg_epoch_loss)
        avg_epoch_acc = epoch_acc / batch_count
        push!(train_accuracies, avg_epoch_acc)

        # VAL ACC

        Flux.testmode!(head_nn)
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        batch_count = 0
        for (batch_idx, real_data) in enumerate(prepared_val_batches)
            prev_layer_mage_outs = vcat([network_batched_val[prev_k][batch_idx][2] for prev_k in prev_layer_keys]...) |> gpu
            parallel_inputs = ntuple(_ -> prev_layer_mage_outs, n_outs)
            real_labels = real_data[2] |> gpu
            outputs = head_nn(parallel_inputs)
            cls_loss = MAGENetwork.classification_loss(outputs, real_labels)
            val_epoch_loss += cls_loss
            batch_count += 1

            # ACC
            preds = Flux.onecold(cpu(outputs))
            true_labels = Flux.onecold(cpu(real_labels))
            correct = sum(preds .== true_labels)
            val_epoch_acc += correct / length(true_labels)
        end
        new_val_acc = val_epoch_acc / batch_count
        Flux.trainmode!(head_nn)
        push!(val_accuracies, new_val_acc)
        if new_val_acc < MAGENetwork.REQ[]
            budget += 1
        end
        new_val = val_accuracies[end]
        if new_val > best_val
            best_val = new_val
            best_at = epoch
            best_weights = deepcopy(Flux.state(head_nn))
        end

        if avg_epoch_loss < best_loss - 0.001
            best_loss = avg_epoch_loss
            epochs_without_improvement = 0
        else
            epochs_without_improvement += 1
        end
        if epoch % 10 == 0
            @info "Epoch $epoch: Loss = $avg_epoch_loss. Best $best_loss. Budget $budget. Val acc : $new_val"
        end
    end
    Flux.loadmodel!(head_nn, best_weights)

    @show minimum(train_losses)
    @show maximum(train_accuracies)
    @show maximum(val_accuracies)

    y_min = min(minimum(train_accuracies), minimum(val_accuracies))
    y_max = max(maximum(train_accuracies), maximum(val_accuracies))
    plt = Plot([NaN], [NaN]; xlim = (1, length(val_accuracies)), ylim = (y_min, y_max), height = 5)
    lineplot!(plt, train_accuracies, name = "train acc")
    lineplot!(plt, val_accuracies, name = "val acc")
    vline!(plt, best_at, color = :yellow, name = "STOP $(best_at)")
    println(plt)


    # HEAD TO CGP
    head_nn = cpu(head_nn)
    for i in 1:n_outs
        c = deepcopy(model[max_k][i].program)
        new_single_genome, out_call_idx = MAGENetwork.nn_to_cgp(head_nn[i], c[1], :relu, model[max_k].ml[1])
        c.genomes[1] = new_single_genome
        c.output_nodes[1][2].value = out_call_idx
        model[max_k][i].program = c # replace prog with nn->cgp
    end

    # MAGENetwork.decode_mn!(model)
    # if train_last_layer
    #     MAGENetwork.train_mage_layer_from_surrogate_data!(
    #         [model[x][y].program for (x, y) in programs_keys],
    #         model[max_k], programs_keys, data, 20, 20
    #     )
    # end
    MAGENetwork.decode_mn!(model)

    # @info length(program_data_nn)
    # P = []
    # for i in program_data_nn
    #     outs = MAGENetwork.forward(model, i.inputs)
    #     push!(P, outs)
    # end
    # new_model_pred_cat = map(i -> argmax(i), P)
    # gt = [i.outputs for i in program_data_nn]
    # acc = mean(new_model_pred_cat .== gt)
    # @info "NEW MODEL ACC : $acc"
end
#

# Fit after
# model = elite_population[1]

mm = MAGENetwork.build_surrogate_model(model)
prepared_val_batches = MAGENetwork.prepare_nn_dataloader(program_data_nn, MAGENetwork.BS[], MAGENetwork.ImagesToScalarNN, :gpu, n_classes; transformations = transformations, shuffle = false)
acc, _ = MAGENetwork.test_parallel_model(
    mm |> gpu,
    prepared_val_batches
)
@info acc
prepared_val_batches = MAGENetwork.prepare_nn_dataloader(val_program_data_nn, MAGENetwork.BS[], MAGENetwork.ImagesToScalarNN, :gpu, n_classes; transformations = transformations, shuffle = false)
acc, _ = MAGENetwork.test_parallel_model(
    mm |> gpu,
    prepared_val_batches
)
@info acc

captured_data = MAGENetwork.with_device(
    model -> begin
        MAGENetwork.capture_surrogate_training_data(
            model, program_data_nn, nothing
        )
    end,
    model,
    :gpu
)
for k in [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)]
    d = Dict(
        "program_instances" => captured_data[k],
        "labels" => [i.outputs for i in program_data_nn]
    )
    save("surrogates_2/surrogate_$(k[1])_$(k[2]).jld2", d)
end


#################################
# TRAIN LOADED MAGENET          #
#################################

device_ = :gpu
n_classes = 10

best_ind_1_1 = deserialize("surrogates/1_1/checkpoint_0.pickle")["best_genome"];
best_ind_1_2 = deserialize("surrogates/1_2/checkpoint_0.pickle")["best_genome"];
best_ind_1_3 = deserialize("surrogates/1_3/checkpoint_0.pickle")["best_genome"];
best_ind_1_4 = deserialize("surrogates/1_4/checkpoint_0.pickle")["best_genome"];
best_ind_1_5 = deserialize("surrogates/1_5/checkpoint_0.pickle")["best_genome"];
best_ind_1_6 = deserialize("surrogates/1_6/checkpoint_0.pickle")["best_genome"];
best_ind_1_7 = deserialize("surrogates/1_7/checkpoint_0.pickle")["best_genome"];
best_ind_1_8 = deserialize("surrogates/1_8/checkpoint_0.pickle")["best_genome"];
best_ind_1_9 = deserialize("surrogates/1_9/checkpoint_0.pickle")["best_genome"];
best_ind_1_10 = deserialize("surrogates/1_10/checkpoint_0.pickle")["best_genome"];

m[1][1].program = best_ind_1_1;
m[1][2].program = best_ind_1_2;
m[1][3].program = best_ind_1_3;
m[1][4].program = best_ind_1_4;
m[1][5].program = best_ind_1_5;
m[1][6].program = best_ind_1_6;
m[1][7].program = best_ind_1_7;
m[1][8].program = best_ind_1_8;
m[1][9].program = best_ind_1_9;
m[1][10].program = best_ind_1_10;

model = m
# elite_population[1] = model
MAGENetwork.decode_mn!(model)

# TRAIN HEAD
batch_size = MAGENetwork.BS[]
updated_train_data = MAGENetwork.collect_training_data(model, map(i -> i.inputs, program_data_nn), true) # this already reflects the updated previous layers
updated_val_data = MAGENetwork.collect_training_data(model, map(i -> i.inputs, val_program_data_nn), true) # this already reflects the updated previous layers
network_structure_train, network_dataloader_train, network_prog_types = MAGENetwork.to_network_structure(updated_train_data, model, batch_size, device_, nothing)
network_structure_val, network_dataloader_val, _ = MAGENetwork.to_network_structure(updated_val_data, model, batch_size, device_, nothing)

ks = keys(network_structure_train)
network_batched_train = OrderedDict()
network_batched_val = OrderedDict()
for k in ks
    network_batched_train[k] = [i for i in network_dataloader_train[k]]
    network_batched_val[k] = [i for i in network_dataloader_val[k]]
end
max_k = maximum(k[1] for k in ks)

prepared_train_batches = MAGENetwork.prepare_nn_dataloader(
    program_data_nn, batch_size, MAGENetwork.ImagesToScalarNN, device_, n_classes,
    onehot = true, shuffle = false, transformations = nothing
) # TO NN format (FIRST INPUT - GROUND TRUTH)
prepared_val_batches = MAGENetwork.prepare_nn_dataloader(
    val_program_data_nn, batch_size, MAGENetwork.ImagesToScalarNN, device_, n_classes;
    onehot = true, shuffle = false, transformations = nothing
) # TO NN format (FIRST INPUT - GROUND TRUTH)
@show length(prepared_train_batches) length(prepared_val_batches)

prev_layer = max_k - 1 # idx prev layer, before head
n_outs_from_prev_layer = model[prev_layer].programs |> length
prev_layer_keys = [(prev_layer, i) for i in 1:n_outs_from_prev_layer]
n_outs = model[max_k].programs |> length
heads = []
for i in 1:n_outs
    nn = Flux.Chain(
        # LayerNorm(n_outs_from_prev_layer),
        BatchNorm(n_outs_from_prev_layer),
        Dense(n_outs_from_prev_layer => 4, relu),
        Dropout(0.1),
        Dense(4 => 3, relu),
        Dropout(0.1),
        Dense(3 => 2, relu),
        Dropout(0.1),
        Dense(2 => 1)
    )
    push!(heads, nn)
end
head_nn = Flux.Parallel(vcat, heads...)
Flux.trainmode!(head_nn)

epoch = 0
budget = 10
head_nn = head_nn |> gpu
optimizer = AdamW(0.001f0)
opt_state = Flux.setup(optimizer, head_nn)
total_loss, train_losses, best_loss = 0.0, Float64[], Inf
epochs_without_improvement = 0
train_accuracies, val_accuracies = Float64[], Float64[]
best_val = -Inf
best_weights = deepcopy(Flux.state(head_nn))
best_at = 0
while epoch < 160 #&& epoch < budget
    epoch += 1
    epoch_loss, batch_count, epoch_acc = 0.0, 0, 0.0
    all_inter_losses_ = 0.0
    for (batch_idx, real_data) in enumerate(prepared_train_batches)
        prev_layer_mage_outs = vcat([network_batched_train[prev_k][batch_idx][2] for prev_k in prev_layer_keys]...) |> gpu
        parallel_inputs = ntuple(_ -> prev_layer_mage_outs, n_outs)
        real_labels = real_data[2] |> gpu
        (loss, losses_align, outputs), grads = Flux.withgradient(head_nn) do m
            outputs = m(parallel_inputs)
            cls_loss = MAGENetwork.classification_loss(outputs, real_labels)
            #reg = snorm(Flux.trainables(m))
            #scaled_reg = λ[] * reg
            final_loss = cls_loss # scaled_reg
            final_loss, (final_loss, cls_loss), outputs
        end
        epoch_loss += loss
        batch_count += 1
        opt_state, parallel_surrogate = Flux.Optimisers.update!(opt_state, head_nn, grads[1])

        # ACC
        preds = Flux.onecold(cpu(outputs))
        true_labels = Flux.onecold(cpu(real_labels))
        correct = sum(preds .== true_labels)
        epoch_acc += correct / length(true_labels)
    end

    avg_epoch_loss = epoch_loss / batch_count
    push!(train_losses, avg_epoch_loss)
    avg_epoch_acc = epoch_acc / batch_count
    push!(train_accuracies, avg_epoch_acc)

    # VAL ACC

    Flux.testmode!(head_nn)
    val_epoch_loss = 0.0
    val_epoch_acc = 0.0
    batch_count = 0
    for (batch_idx, real_data) in enumerate(prepared_val_batches)
        prev_layer_mage_outs = vcat([network_batched_val[prev_k][batch_idx][2] for prev_k in prev_layer_keys]...) |> gpu
        parallel_inputs = ntuple(_ -> prev_layer_mage_outs, n_outs)
        real_labels = real_data[2] |> gpu
        outputs = head_nn(parallel_inputs)
        cls_loss = MAGENetwork.classification_loss(outputs, real_labels)
        val_epoch_loss += cls_loss
        batch_count += 1

        # ACC
        preds = Flux.onecold(cpu(outputs))
        true_labels = Flux.onecold(cpu(real_labels))
        correct = sum(preds .== true_labels)
        val_epoch_acc += correct / length(true_labels)
    end
    new_val_acc = val_epoch_acc / batch_count
    Flux.trainmode!(head_nn)
    push!(val_accuracies, new_val_acc)
    if new_val_acc < MAGENetwork.REQ[]
        budget += 1
    end
    new_val = val_accuracies[end]
    if new_val > best_val
        best_val = new_val
        best_at = epoch
        best_weights = deepcopy(Flux.state(head_nn))
    end

    if avg_epoch_loss < best_loss - 0.001
        best_loss = avg_epoch_loss
        epochs_without_improvement = 0
    else
        epochs_without_improvement += 1
    end
    if epoch % 10 == 0
        @info "Epoch $epoch: Loss = $avg_epoch_loss. Best $best_loss. Budget $budget. Val acc : $new_val"
    end
end
Flux.loadmodel!(head_nn, best_weights)

@show minimum(train_losses)
@show maximum(train_accuracies)
@show maximum(val_accuracies)

y_min = min(minimum(train_accuracies), minimum(val_accuracies))
y_max = max(maximum(train_accuracies), maximum(val_accuracies))
plt = Plot([NaN], [NaN]; xlim = (1, length(val_accuracies)), ylim = (y_min, y_max), height = 5)
lineplot!(plt, train_accuracies, name = "train acc")
lineplot!(plt, val_accuracies, name = "val acc")
vline!(plt, best_at, color = :yellow, name = "STOP $(best_at)")
println(plt)


# HEAD TO CGP
head_nn = cpu(head_nn)
for i in 1:n_outs
    c = deepcopy(model[max_k][i].program)
    new_single_genome, out_call_idx = MAGENetwork.nn_to_cgp(head_nn[i], c[1], :relu, model[max_k].ml[1])
    c.genomes[1] = new_single_genome
    c.output_nodes[1][2].value = out_call_idx
    model[max_k][i].program = c # replace prog with nn->cgp
end

# MAGENetwork.decode_mn!(model)
# if train_last_layer
#     MAGENetwork.train_mage_layer_from_surrogate_data!(
#         [model[x][y].program for (x, y) in programs_keys],
#         model[max_k], programs_keys, data, 20, 20
#     )
# end
MAGENetwork.decode_mn!(model)

@info length(program_data_nn)
P = []
for i in program_data_nn
    outs = MAGENetwork.forward(model, i.inputs)
    push!(P, outs)
end
new_model_pred_cat = map(i -> argmax(i), P)
gt = [i.outputs for i in program_data_nn]
acc = mean(new_model_pred_cat .== gt)
@info "NEW MODEL ACC : $acc"


# Example of saving mnist data as CLASSIFICATION_DATASET_IMG_SCALAR
# (trainx, trainy, valx, valy, testx, testy, allx, ally) = load_mnist_dataset(0.8);

# all_train_data = CLASSIFICATION_DATASET_IMG_SCALAR(
#     [trainx..., valx...],
#     map(x -> x[1], [trainy..., valy...]),
#     Dict()
# )

# save_object(
#     "datasets_pickle/mnist_train.jld2",
#     all_train_data
# )

# test_data = CLASSIFICATION_DATASET_IMG_SCALAR(
#     testx,
#     map(x -> x[1], testy),
#     Dict()
# )

# save_object(
#     "datasets_pickle/mnist_test.jld2",
#     test_data
# )

# EXAMPLE OF SAVING HF LF DLBCL TO CLASSIFCATION_DATASET_IMG_SCALAR
S = (224, 224)
(trainx, trainy, valx, valy, testx, testy, allx, ally, extras) = load_hf_lf_dlbcl_dataset("datasets", "patches_camilo", "annotation_evostar.csv"; resize_to = S);
train_data = CLASSIFICATION_DATASET_IMG_SCALAR(
    trainx, trainy,
    Dict(
        "metadata" => extras["train_metadata"],
        "labels" => extras["unique_labels"],
        "labels_to_int" => extras["label_to_int"]
    )
)
save_object(
    "datasets_pickle/HFLFDLBCL_train_$(S[1]).jld2",
    train_data
)
val_data = CLASSIFICATION_DATASET_IMG_SCALAR(
    valx, valy,
    Dict(
        "metadata" => extras["val_metadata"],
        "labels" => extras["unique_labels"],
        "labels_to_int" => extras["label_to_int"]
    )
)
save_object(
    "datasets_pickle/HFLFDLBCL_val_$(S[1]).jld2",
    val_data
)
test_data = CLASSIFICATION_DATASET_IMG_SCALAR(
    testx, testy,
    Dict(
        "metadata" => extras["test_metadata"],
        "labels" => extras["unique_labels"],
        "labels_to_int" => extras["label_to_int"]
    )
)
save_object(
    "datasets_pickle/HFLFDLBCL_test_$(S[1]).jld2",
    test_data
)

# SAVING CIFAR DATA ###################################
(trainx, trainy, valx, valy, testx, testy, allx, ally, extras) = load_cifar_dataset(0.8)

for (x, y, subset) in [(trainx, trainy, "train"), (valx, valy, "val"), (testx, testy, "test")]
    tmp = (
        xs = [[reinterpret.(UInt8, c.img) for c in i] for i in x],
        ys = y,
        extras = (
            :struct_ => "CLASSIFICATION_DATASET_IMG_SCALAR",
        ),
    )
    save_object(
        "datasets_pickle/cifar10_$(subset).jld2",
        tmp
    )
end

# Fashion MNIST ###################################
(trainx, trainy, valx, valy, testx, testy, allx, ally, extras) = load_FashionMNIST_dataset(0.8)
train_data = CLASSIFICATION_DATASET_IMG_SCALAR(
    trainx, trainy,
    Dict()
)
save_object(
    "datasets_pickle/fashion_train.jld2",
    train_data
)

val_data = CLASSIFICATION_DATASET_IMG_SCALAR(
    valx, valy,
    Dict()
)
save_object(
    "datasets_pickle/fashion_val.jld2",
    val_data
)

test_data = CLASSIFICATION_DATASET_IMG_SCALAR(
    testx, testy,
    Dict()
)
save_object(
    "datasets_pickle/fashion_test.jld2",
    test_data
)


######################################## SAVING PCAM DATASET #############################################

# RGB ONLY
(trainx, trainy, valx, valy, testx, testy, allx, ally, extras) = load_pcam_dataset("datasets/PCAM/Images", false)
trainx = [x[1:3] for x in trainx];
valx = [x[1:3] for x in valx];
testx = [x[1:3] for x in testx];

# save for julia & python in UInt8
for ((x, y), subset) in zip(
        ((trainx, trainy), (valx, valy), (testx, testy)),
        ("train", "val", "test"),
    )
    @assert length(x) == length(y)
    @info "Subset has : $(length(x)) instances"
    tmp = (
        # x, y,
        xs = [[reinterpret.(UInt8, c.img) for c in i] for i in x],
        ys = y,
        extras = (
            files = extras[Symbol("$(subset)_files")],
            struct_ = "CLASSIFICATION_DATASET_IMG_SCALAR",
        ),
    )
    save_object(
        "datasets_pickle/pcam_rgb_$(subset).jld2",
        tmp
    )
end


# NOISE
include(joinpath(home, "utils", "noise.jl"))

######################################## SAVE PCAM WITH NOISE X (0.01, 0.05, 0.1, 0.2) #############################################

for noise_level in 0.01:0.01:0.2
    noise_level = noise_level * 255
    @info "Making data with noise $noise_level"
    d = Normal(0.0, noise_level)
    # rng = Xoshiro(0)
    # train_x_noise = [add_noise(x, d, rng) for x in trainx]
    rng = Xoshiro(0)
    val_x_noise = [add_noise(x, d, rng) for x in valx]
    rng = Xoshiro(0)
    test_x_noise = [add_noise(x, d, rng) for x in testx]
    for ((x, y), subset) in zip(
            (
                # (train_x_noise, trainy),
                (val_x_noise, valy),
                (test_x_noise, testy),
            ),
            (
                # "train",
                "val", "test",
            ),
        )
        @assert length(x) == length(y)
        @info "Subset has : $(length(x)) instances"
        noise_level_str = replace(string(round(noise_level / 255; digits = 2)), "." => "_")
        save_object(
            "datasets_pickle/pcam_guassian_noise/pcam_rgb_noise_$(noise_level_str)_$(subset).jld2",
            (
                xs = x,
                ys = y,
                extras = (
                    files = extras[Symbol("$(subset)_files")],
                    struct_ = "CLASSIFICATION_DATASET_IMG_SCALAR",
                ),
            )
        )
    end
end

######################################## POISSON RANDOM 100, 255, 1000, 2000#############################################

for photon_level in [2000, 1000, 500, 255, 100, 20, 10]
    @info "Making data with noise (expected photons) $photon_level"
    # rng = Xoshiro(0)
    # train_x_noise = [sample_poisson_noise([float64.(x.img) for x in img], photon_level, rng) for img in trainx]
    rng = Xoshiro(0)
    val_x_noise = [sample_poisson_noise([float64.(x.img) for x in img], photon_level, rng) for img in valx]
    rng = Xoshiro(0)
    test_x_noise = [sample_poisson_noise([float64.(x.img) for x in img], photon_level, rng) for img in testx]
    for ((x, y), subset) in zip(
            (
                # (train_x_noise, trainy),
                (val_x_noise, valy), (test_x_noise, testy),
            ),
            (
                # "train",
                "val", "test",
            ),
        )
        @assert length(x) == length(y)
        @info "Subset has : $(length(x)) instances"
        save_object(
            "datasets_pickle/pcam_poisson_noise/pcam_rgb_poisson_$(photon_level)_$(subset).jld2",
            (
                xs = x,
                ys = y,
                extras = (
                    files = extras[Symbol("$(subset)_files")],
                    struct_ = "CLASSIFICATION_DATASET_IMG_SCALAR",
                ),
            )
        )
    end
end


######################################## BRIGHTNESS -0.3, -0.2, -0.15, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3 #############################################
for brightness_level in [-0.3, -0.2, -0.15, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3] # only do for test
    @info "Adding brightness $brightness_level "
    # train_x_noise = [add_brightness([float64.(x.img) for x in img], brightness_level) for img in trainx]
    val_x_noise = [add_brightness([float64.(x.img) for x in img], brightness_level) for img in valx]
    test_x_noise = [add_brightness([float64.(x.img) for x in img], brightness_level) for img in testx]
    for ((x, y), subset) in zip(
            (
                # (train_x_noise, trainy),
                (val_x_noise, valy), (test_x_noise, testy),
            ),
            (
                # "train",
                "val", "test",
            ),

        )
        @assert length(x) == length(y)
        @info "Subset has : $(length(x)) instances"
        br = abs(brightness_level)
        level_str = replace(string(br), "." => "_")
        level_str = brightness_level < 0 ? "neg_$level_str" : "pos_$level_str"
        save_object(
            "datasets_pickle/pcam_brigthness_noise/pcam_rgb_brightness_$(level_str)_$(subset).jld2",
            (
                xs = x,
                ys = y,
                extras = (
                    files = extras[Symbol("$(subset)_files")],
                    struct_ = "CLASSIFICATION_DATASET_IMG_SCALAR",
                ),
            )
        )
    end
end


# HFLF ################################################################

# HF LF DATA 20/10/25 ###################################
folder = "datasets_pickle"
S1 = 32
folder = joinpath(folder, "hflf_$S1")
isdir(folder) || mkdir(folder)
S = (S1, S1)
S_str = "$(S[1])_$(S[2])"
BASE_NAME = "HFLF_rgb"

(trainx, trainy, valx, valy, testx, testy, allx, ally, extras) = load_hf_lf_dataset("datasets/patches_camilo_new"; resize_to = S);
trainx = [x[1:3] for x in trainx];
valx = [x[1:3] for x in valx];
testx = [x[1:3] for x in testx];

for (x, y, subset) in [(trainx, trainy, "train"), (valx, valy, "val"), (testx, testy, "test")]
    tmp = (
        xs = [[reinterpret.(UInt8, c.img) for c in i] for i in x],
        ys = y,
        extras = (
            metadata = extras["$(subset)_metadata"],
            labels = extras["unique_labels"],
            labels_to_int = extras["label_to_int"],
            struct_ = "CLASSIFICATION_DATASET_IMG_SCALAR",
        ),
    )
    save_object(
        "$folder/$(BASE_NAME)_$(subset)_$(S_str).jld2",
        tmp
    )
end
for noise_level in 0.01:0.01:0.2
    noise_level = noise_level * 255
    @info "Making data with noise $noise_level"
    d = Normal(0.0, noise_level)
    # rng = Xoshiro(0)
    # train_x_noise = [add_noise(x, d, rng) for x in trainx]
    rng = Xoshiro(0)
    val_x_noise = [add_noise(x, d, rng) for x in valx]
    rng = Xoshiro(0)
    test_x_noise = [add_noise(x, d, rng) for x in testx]
    for ((x, y), subset) in zip(
            (
                # (train_x_noise, trainy),
                (val_x_noise, valy),
                (test_x_noise, testy),
            ),
            (
                # "train",
                "val",
                "test",
            ),
        )
        @assert length(x) == length(y)
        @info "Subset has : $(length(x)) instances"
        noise_level_str = replace(string(round(noise_level / 255; digits = 2)), "." => "_")
        save_object(
            "$folder/$(BASE_NAME)_noise_$(noise_level_str)_$(subset).jld2",
            (
                xs = x,
                ys = y,
                extras = (
                    metadata = extras["$(subset)_metadata"],
                    labels = extras["unique_labels"],
                    labels_to_int = extras["label_to_int"],
                    struct_ = "CLASSIFICATION_DATASET_IMG_SCALAR",
                ),
            )
        )
    end
end
for photon_level in [2000, 1000, 500, 255, 100, 20, 10]
    @info "Making data with noise (expected photons) $photon_level"
    # rng = Xoshiro(0)
    # train_x_noise = [sample_poisson_noise([float64.(x.img) for x in img], photon_level, rng) for img in trainx]
    rng = Xoshiro(0)
    val_x_noise = [sample_poisson_noise([float64.(x.img) for x in img], photon_level, rng) for img in valx]
    rng = Xoshiro(0)
    test_x_noise = [sample_poisson_noise([float64.(x.img) for x in img], photon_level, rng) for img in testx]
    for ((x, y), subset) in zip(
            (
                # (train_x_noise, trainy),
                (val_x_noise, valy),
                (test_x_noise, testy),
            ),
            (
                # "train",
                "val",
                "test",
            ),
        )
        @assert length(x) == length(y)
        @info "Subset has : $(length(x)) instances"
        save_object(
            "$folder/$(BASE_NAME)_poisson_$(photon_level)_$(subset).jld2",
            (
                xs = x,
                ys = y,
                extras = (
                    metadata = extras["$(subset)_metadata"],
                    labels = extras["unique_labels"],
                    labels_to_int = extras["label_to_int"],
                    struct_ = "CLASSIFICATION_DATASET_IMG_SCALAR",
                ),
            )
        )
    end
end
for brightness_level in [-0.3, -0.2, -0.15, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3] # only do for test
    @info "Adding brightness $brightness_level "
    # train_x_noise = [add_brightness([float64.(x.img) for x in img], brightness_level) for img in trainx]
    val_x_noise = [add_brightness([float64.(x.img) for x in img], brightness_level) for img in valx]
    test_x_noise = [add_brightness([float64.(x.img) for x in img], brightness_level) for img in testx]
    for ((x, y), subset) in zip(
            (
                # (train_x_noise, trainy),
                (val_x_noise, valy),
                (test_x_noise, testy),
            ),
            (
                # "train",
                "val",
                "test",
            ),
        )
        @assert length(x) == length(y)
        @info "Subset has : $(length(x)) instances"
        br = abs(brightness_level)
        level_str = replace(string(br), "." => "_")
        level_str = brightness_level < 0 ? "neg_$level_str" : "pos_$level_str"
        save_object(
            "$folder/$(BASE_NAME)_brightness_$(level_str)_$(subset).jld2",
            (
                xs = x,
                ys = y,
                extras = (
                    metadata = extras["$(subset)_metadata"],
                    labels = extras["unique_labels"],
                    labels_to_int = extras["label_to_int"],
                    struct_ = "CLASSIFICATION_DATASET_IMG_SCALAR",
                ),
            )
        )
    end
end
