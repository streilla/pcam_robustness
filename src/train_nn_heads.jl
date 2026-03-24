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
pinthreads(:cores) ; println(threadinfo(; slurm = true, color = false))

setup_magenetwork_constants(
    TH_FOR_ALIGN, TH_FOR_ES, FREEZE_RATE, BATCH_SIZE, REGULARIZATION, INTER_LOSSES, DROPOUT_RATE, RESET_PB, OPTIM, ACT, MAX
)

const _device = setup_cuda_device(DEVICE_CU)
include_utils_and_disable_logging(home)

Parsed_args["n_elite"] = 1; @info "Setting n_elite to 1 since not needed"

######################
# READ FILES #########
######################

path_to_data = Parsed_args["data_location"]
all_train_data = load(path_to_data)["single_stored_object"]
(trainx, trainy), (valx, valy) = MLUtils.splitobs((all_train_data.xs, all_train_data.ys); at = 0.8, shuffle = true, stratified = all_train_data.ys)

N_CLASSES = length(unique(valy))
only_float_bundles = UTCGP.get_sr_float_bundles()
set_bundle_casters!(only_float_bundles, float_caster2)
ml_float = ml_from_vbundles([only_float_bundles])

#################################
# TRAIN ONLY THE HEADS          #
#################################

device_ = :gpu
BS = 512
train_distill_instances = MAGENetwork.MAGEProgramInstance[]
for (x, y) in zip(trainx, trainy)
    push!(train_distill_instances, MAGENetwork.MAGEProgramInstance(x, y))
end
val_distill_instances = MAGENetwork.MAGEProgramInstance[]
for (x, y) in zip(valx, valy)
    push!(val_distill_instances, MAGENetwork.MAGEProgramInstance(x, y))
end

prepared_train_batches = MAGENetwork.prepare_nn_dataloader(
    train_distill_instances, BS, MAGENetwork.ScalarsToScalarNN, device_, N_CLASSES,
    onehot = true, shuffle = false, transformations = nothing
) # TO NN format (FIRST INPUT - GROUND TRUTH)
prepared_val_batches = MAGENetwork.prepare_nn_dataloader(
    val_distill_instances, BS, MAGENetwork.ScalarsToScalarNN, device_, N_CLASSES;
    onehot = true, shuffle = false, transformations = nothing
) # TO NN format (FIRST INPUT - GROUND TRUTH)
@show length(prepared_train_batches) length(prepared_val_batches)


n_outs = length(trainx[1])
heads = []
for i in 1:N_CLASSES
    nn = Flux.Chain(
        Dense(n_outs => 10; init = Flux.kaiming_uniform),
        BatchNorm(10),
        Flux.relu,
        Dropout(0.2),

        Dense(4 => 3, relu; init = Flux.kaiming_uniform),
        Dropout(0.2),
        Dense(3 => 2, relu; init = Flux.kaiming_uniform),
        Dropout(0.2),
        Dense(2 => 1; init = Flux.kaiming_uniform)
    )
    push!(heads, nn)
end
head_nn = Flux.Parallel(vcat, heads...)
Flux.trainmode!(head_nn)

epoch = 0
budget = 10
head_nn = head_nn |> gpu
optimizer = AdamW(0.001f0)
optimizer = OptimiserChain(ClipGrad(0.5), optimizer)
opt_state = Flux.setup(optimizer, head_nn)
total_loss, train_losses, best_loss = 0.0, Float64[], Inf
epochs_without_improvement = 0
train_accuracies, val_accuracies = Float64[], Float64[]
best_val = -Inf
best_weights = deepcopy(Flux.state(head_nn))
best_at = 0
while epoch < 100
    epoch += 1
    epoch_loss, batch_count, epoch_acc = 0.0, 0, 0.0
    all_inter_losses_ = 0.0
    for (batch_idx, real_data) in enumerate(prepared_train_batches)
        x, y = real_data
        x_gpu, y_gpu = x |> gpu, y |> gpu
        parallel_inputs = ntuple(_ -> x_gpu, N_CLASSES)
        (loss, losses_align, outputs), grads = Flux.withgradient(head_nn) do m
            outputs = m(parallel_inputs)
            cls_loss = MAGENetwork.classification_loss(outputs, y_gpu)
            final_loss = cls_loss # scaled_reg
            final_loss, (final_loss, cls_loss), outputs
        end
        epoch_loss += loss
        batch_count += 1
        opt_state, parallel_surrogate = Flux.Optimisers.update!(opt_state, head_nn, grads[1])

        # ACC
        preds = Flux.onecold(cpu(outputs))
        true_labels = Flux.onecold(y)
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
        x, y = real_data
        x_gpu, y_gpu = x |> gpu, y |> gpu
        parallel_inputs = ntuple(_ -> x_gpu, N_CLASSES)
        outputs = head_nn(parallel_inputs)
        cls_loss = MAGENetwork.classification_loss(outputs, y_gpu)
        val_epoch_loss += cls_loss
        batch_count += 1

        # ACC
        preds = Flux.onecold(cpu(outputs))
        true_labels = Flux.onecold(y)
        correct = sum(preds .== true_labels)
        val_epoch_acc += correct / length(true_labels)
    end
    new_val_acc = val_epoch_acc / batch_count
    Flux.trainmode!(head_nn)
    push!(val_accuracies, new_val_acc)

    new_val = val_accuracies[end]
    if new_val > best_val
        best_val = new_val
        best_at = epoch
        best_weights = deepcopy(Flux.state(head_nn))
    end

    if epoch % 10 == 0
        @info "Epoch $epoch: Loss = $avg_epoch_loss. Val acc : $new_val"
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

# Save results (metrics)
head_nn = cpu(head_nn)
Flux.testmode!(head_nn)
payload = Dict(
    "train_losses" => train_losses,
    "train_accuracies" => train_accuracies,
    "val_accuracies" => val_accuracies
)
folder = joinpath(Parsed_args["output_dir"], Parsed_args["trial_id"])
payload_path = joinpath(folder, "nn_heads")
isdir(payload_path) || mkdir(payload_path)
open(joinpath(payload_path, "ann_heads_results.json"), "w") do f
    JSON.print(f, payload)
end

# Save results (ANNs)
JLD2.save_object(
    joinpath(payload_path, "heads.jld2"),
    Dict(
        "state" => Flux.state(head_nn),
        "n_outs" => n_outs,
        "n_classes" => N_CLASSES,
        "model_example" => head_nn[1]
    )
)

# CONVERT HEADS TO CGP
model_arch = modelArchitecture(
    [Float64 for i in 1:n_outs],
    [1 for i in 1:n_outs],
    [Float64],
    [Float64],
    [1]
)
node_config = nodeConfig(2000, 1, 3, n_outs)
shared_in, super_large_genome = make_evolvable_utgenome(
    model_arch, ml_float, node_config
)
initialize_genome!(super_large_genome)
correct_all_nodes!(super_large_genome, model_arch, ml_float, shared_in)
new_single_genome, out_call_idx = MAGENetwork.nn_to_cgp(head_nn[1], super_large_genome[1], :relu, ml_float[1])
# decode
super_large_genome.genomes[1] = new_single_genome
super_large_genome.output_nodes[1][2].value = out_call_idx
decoded = UTCGP.decode_with_output_nodes(super_large_genome, ml_float, model_arch, shared_in)

out_call_idx += 1
@info "To fit the ANN head in CGP we need $out_call_idx nodes"

ANNS_TO_GENOMES = []
for i in 1:N_CLASSES
    global out_call_idx
    node_config = nodeConfig(out_call_idx, 1, 3, n_outs)
    shared_in, genome = make_evolvable_utgenome(
        model_arch, ml_float, node_config
    )
    initialize_genome!(genome)
    correct_all_nodes!(genome, model_arch, ml_float, shared_in)
    new_single_genome, out = MAGENetwork.nn_to_cgp(head_nn[i], genome[1], :relu, ml_float[1])
    genome.genomes[1] = new_single_genome
    genome.output_nodes[1][2].value = out
    push!(ANNS_TO_GENOMES, genome)
end

JLD2.save_object(
    joinpath(payload_path, "cgp_heads.jld2"),
    ANNS_TO_GENOMES
)


# test if the outputs match between cgp and heads

# Compare one NN and its CGP translation on a single input
function compare_nn_cgp(nn::Chain, program, x::AbstractVector{<:Real}, arch, ml_float)
    n = length(x)
    # Run through the ANN
    y_nn = nn(reshape(Float32.(x), (n, 1)))

    # Run through the CGP
    UTCGP.reset_program!.(program)
    UTCGP.replace_shared_inputs!(program, x)
    y_cgp = UTCGP.evaluate_individual_programs(program, arch.chromosomes_types, ml_float)
    # y_cgp = INTER_ACT(y_cgp)
    return y_nn, y_cgp
end

# Compare all pairs in ANNS_TO_GENOMES vs head_nn
function check_all_equivalence(head_nn, ANNS_TO_GENOMES, x::AbstractVector, arch, ml_float; atol = 1.0e-6)
    programs = [UTCGP.decode_with_output_nodes(g, ml_float, model_arch, shared_in) for g in ANNS_TO_GENOMES]
    results_modules = []
    for (i, (nn, program)) in enumerate(zip(head_nn.layers, programs))
        results = Float64[]
        for x in trainx
            y_nn, y_cgp = compare_nn_cgp(nn, program, x, arch, ml_float)
            diff = maximum(abs.(y_nn .- y_cgp))
            push!(results, diff)
        end
        push!(results_modules, results)
    end
    return results_modules
end

# Example usage on first training sample
res = check_all_equivalence(head_nn, ANNS_TO_GENOMES, trainx[1:1000], model_arch, ml_float)
mean.(res)
