st = time()

using Base.Threads
using LinearAlgebra
using InteractiveUtils
using BenchmarkTools
import JSON
@show pwd()
nt = nthreads()
@show nt
@show BLAS.get_num_threads()
BLAS.set_num_threads(1)
@show BLAS.get_num_threads()
@show Sys.CPU_THREADS
using HypothesisTests
using Revise
using Dates
using Infiltrator
using Debugger
using UTCGP
import SearchNetworks as sn
import DataStructures: OrderedDict
using UUIDs
using Images, FileIO
using ImageCore
using ImageBinarization
using CSV
using DataFrames
using Statistics
using Random
using StatsBase: sample
using Flux
using Logging
using HDF5
using DataFlowTasks
using ArgParse
import PNGFiles
using ThreadPinning
using MLDatasets
using MLDataUtils
using MAGENetwork

pinthreads(:cores)
ENV["JULIA_DEBUG"] = "UTCGP"
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))
if occursin("REPL", @__FILE__)
    # Parsed_args = Dict("seed" => 1,
    #     "mutation_rate" => 1,
    #     "n_nodes" => 10,
    #     "n_elite" => 6,
    #     "n_new" => 14,
    #     "tour_size" => 3,
    #     "n_samples" => 97,
    #     "n_repetitions" => 3,
    #     "budget" => 200_000,
    #     "acc_weight" => 0.51,
    #     "instance" => "repl",
    # )
    global home
    home = "./"
else
    Parsed_args = parse_args(s)
end

Random.seed!(1)
ACC_WEIGHT = 0.8
NSAMPLES = 100
# Settings
include(joinpath(home, "utils/utils.jl"))
disable_logging(Logging.Debug)
hash = UUIDs.uuid4() |> string

# MNIST DATASET 
mnist_test = MLDatasets.MNIST.testdata()
example = mnist_test[1][:, :, 1]
IMGTYPE = typeof(SImageND(example))

# DATA SPLIT  --- 
mnist_train = MLDatasets.MNIST.traindata()
mnist_train_y = mnist_train[2]
mnist_test = MLDatasets.MNIST.testdata()
mnist_test_y = mnist_test[2]

mnist_train = [[SImageND(mnist_train[1][:, :, i]), 0.1, 0.2, 2.0, 3.0, -0.1,] for i in 1:size(mnist_train[1])[3]]
mnist_test = [[SImageND(mnist_test[1][:, :, i])] for i in 1:size(mnist_test[1])[3]]

(mnist_train_x, mnist_train_y), (mnist_val_x, mnist_val_y) = splitobs((mnist_train, mnist_train_y); at=0.7)
mnist_train_y = [[i] for i in mnist_train_y]
mnist_val_y = [[i] for i in mnist_val_y]

@show size(mnist_train_x)
@show size(mnist_val_x)

VALDataloader = RamDataLoader(mnist_val_x, mnist_val_y,
    collect(1:length(mnist_val_x)),
    ceil(Int, NSAMPLES / nt), NSAMPLES)

TRAINDataloader = RamDataLoader(mnist_train_x, mnist_train_y, collect(1:length(mnist_train_y)), ceil(Int, NSAMPLES / nt), NSAMPLES)

# MNIST DATASET 
example = mnist_train_x[1][1]
IMGTYPE = typeof(example)
NOUTS = length(unique(mnist_train_y))

# NETWORK 
ma_image2image = modelArchitecture(
    [IMGTYPE, Float64, Float64, Float64, Float64, Float64],
    [1, 2, 2, 2, 2, 2],
    [IMGTYPE, Float64],
    [IMGTYPE],
    [1]
)
ma_image2float = modelArchitecture(
    [IMGTYPE for i in 1:NOUTS],
    [1 for i in 1:NOUTS],
    [IMGTYPE, Float64],
    [IMGTYPE],
    [2]
)
fallback() = SImageND(ones(N0f8, size(example)))
image2D = UTCGP.get_image2D_factory_bundles()
push!(image2D, UTCGP.experimental_bundle_image2D_mask_factory)
for factory_bundle in image2D
    for (i, wrapper) in enumerate(factory_bundle)
        fn = wrapper.fn(IMGTYPE) # specialize
        wrapper.fallback = fallback
        factory_bundle.functions[i] =
            UTCGP.FunctionWrapper(fn, wrapper.name, wrapper.caster, wrapper.fallback)
    end
end
float_bundles = UTCGP.get_float_bundles()
lib_image2D = Library(image2D)
lib_float = Library(float_bundles)
ml = MetaLibrary([lib_image2D, lib_float])
pnc = prenodeConfig(3, 3)

initial_pop = MNModel[]
pop_size = 5
for i in 1:pop_size
    l1 = MNLayer(10, ma_image2image, ml, pnc)
    l2 = MNLayer(10, ma_image2float, ml, pnc)
    mn_model = MNModel(
        MNSequence(
            l1,
            l2
        ))
    init_mn!(mn_model)
    push!(initial_pop, mn_model)
end
initial_pop = MNPopulation(initial_pop);

n_repetitions = 1
rc = MNRunConf(
    ; gens=10, n_elite=5, n_new=2, ts=2,
    mutation_n_models=2,
    mutation_model=1
)

struct MNIST_ENDPOINT <: UTCGP.BatchEndpoint
    fitness_results::Vector{Float64}
    function MNIST_ENDPOINT(
        pop_preds::Vector{<:Vector{<:Number}}, # pop[ ind1[ out1, out2 ], ind2... ]. 
        truth::Vector{<:Int}, # one obs
    )
        global ACC_WEIGHT
        l = unique(length.(pop_preds))[1]
        pop_res = Float64[]
        for ind_predictions in pop_preds
            ind_predictions = replace(ind_predictions, NaN => 0.0)
            pred = argmax(ind_predictions)
            gt = truth[1]
            one_hot_gt = collect(1:l) .== gt
            nll_loss = Flux.logitcrossentropy(
                ind_predictions,
                one_hot_gt,
                dims=1)
            nll_loss = clamp(nll_loss, 0.0, 1.0)
            acc = pred == gt ? 0.0 : 1.0
            push!(pop_res, ACC_WEIGHT * acc + (1 - ACC_WEIGHT) * nll_loss)
        end
        return new(pop_res)
    end
end

fit_ga_network_batch(
    TRAINDataloader,
    nothing,
    n_repetitions,
    initial_pop,
    rc,
    nothing,
    # Callbacks before step (before looping through data)
    (mnga_pop_callback,),
    (mnga_mutation_callback!,),
    # output_mutation_callbacks::UTCGP.Mandatory_FN,
    (mnga_decoding_callback!,),
    # Callbacks per step (while looping through data)
    MNIST_ENDPOINT,
    nothing, #final_step_callbacks::UTCGP.Optional_FN,
    # Callbacks after step ::
    (mnga_selection_callback,),
    nothing, #epoch_callbacks::UTCGP.Optional_FN,
    nothing, #early_stop_callbacks::UTCGP.Optional_FN,
    nothing #last_callback::UTCGP.Optional_FN,
)

# PARAMS --- --- 
# endpoint = PCAMEndpoint

# TRACKING
# h_params = Dict(
#     "connection_temperature" => node_config.connection_temperature,
#     "n_nodes" => node_config.n_nodes,
#     "generations" => run_conf.generations,
#     "budget" => Parsed_args["budget"],
#     "mutation_rate" => run_conf.mutation_rate,
#     "output_mutation_rate" => run_conf.output_mutation_rate,
#     "Correction" => "true",
#     "output_node" => "fixed",
#     "n_train" => Parsed_args["n_samples"],
#     "n_test" => length(valx),
#     "seed" => Parsed_args["seed"],
#     "mutation" => "ga_numbered_new_material_mutation_callback",
#     "n_elite" => Parsed_args["n_elite"],
#     "n_new" => Parsed_args["n_new"],
#     "tour_size" => Parsed_args["tour_size"],
#     "acc_weight" => ACC_WEIGHT,
#     "instance" => Parsed_args["instance"],
#     "n_repetitions" => N_REPETITIONS
# )

#######
# FIT #
#######
# struct acc_callback <: AbstractCallable
# X_test
# Y_test
# subset
# end

# struct Ensemble
#     acc_data::acc_callback
# end

# function (e::Ensemble)(pop::PopulationPrograms, model_arch::modelArchitecture, meta_library::MetaLibrary)
#     UTCGP.reset_programs!.(pop)
#     n = length(e.acc_data.X_test)
#     losses_avg = Vector{Float64}(undef, n) # avg
#     losses_median = Vector{Float64}(undef, n) # avg
#     n_progs = length(pop)
#     nt = Threads.nthreads()
#     Thread_Size = ceil(Int, n_progs / nt)
#     OUTPUTS = Vector(undef, n_progs)
#     xs, ys = e.acc_data.X_test[1:n], e.acc_data.Y_test[1:n]
#     for ith_progs in Iterators.partition(1:n_progs, Thread_Size)
#         @info "Gave the task to fit progs $ith_progs"
#         # unpack input nodes
#         slots = @view OUTPUTS[ith_progs]
#         @dspawn begin
#             for (i, prog_idx) in enumerate(ith_progs)
#                 @info "Fitting prog $prog_idx at $(Threads.threadid())"
#                 @W slots
#                 prog_outputs = []
#                 prog = pop[prog_idx]
#                 non_shared_programs = deepcopy(prog)
#                 in_types = model_arch.inputs_types_idx
#                 fitnesses = Float64[]
#                 # append input nodes to pop
#     for (x, y) in zip(xs, ys)
#         UTCGP.reset_programs!(non_shared_programs)
#         input_nodes = [
#             InputNode(value, pos, pos, in_types[pos]) for
#             (pos, value) in enumerate(x)
#         ]
#         replace_shared_inputs!(non_shared_programs, input_nodes)
#         outputs = UTCGP.evaluate_individual_programs(
#             non_shared_programs,
#             model_arch.chromosomes_types,
#             meta_library,
#         )
#         # Endpoint results
#         # fitness = argmax(outputs) == y[1] ? 1.0 : 0.0
#         # push!(fitnesses, fitness)
#         push!(prog_outputs, outputs)
#     end
#     slots[i] = prog_outputs
# end
#         end
#     end
#     final_task = @dspawn @R(@view OUTPUTS[begin:end]) label = "result"
#     fetch(final_task)
#     PREDS = Matrix{Int}(undef, n_progs, n)
#     for (i, ind) in enumerate(OUTPUTS)
#         for (j, pred) in enumerate(ind)
#             PREDS[i, j] = argmax(pred)
#         end
#     end
#     final_preds_avg = round.(mean(PREDS, dims=1))
#     final_preds_median = round.(median(PREDS, dims=1))
#     for i in 1:n
#         losses_avg[i] = final_preds_avg[i] == ys[i][1] ? 1.0 : 0.0
#         losses_median[i] = final_preds_median[i] == ys[i][1] ? 1.0 : 0.0
#     end
#     # acc = count(x -> x == 1.0, losses) / length(losses)
#     # println("$(a.subset) Accuracy : $(round(acc,digits = 2))")
#     # acc
#     OUTPUTS, PREDS, final_preds_avg, final_preds_median, losses_avg, losses_median
# end

# function (a::acc_callback)(
#     ind_performances,
#     population,
#     iteration,
#     run_config,
#     model_architecture,
#     node_config,
#     meta_library,
#     shared_inputs,
#     population_programs,
#     best_losses,
#     best_programs,
#     elite_idx,
#     batch
# )
#     use_batch = length(a.X_test) == 0
#     use_batch ? n = length(batch) : n = length(a.X_test)
#     UTCGP.reset_programs!.(best_programs)
#     best_ind_idx = argmin(best_losses)
#     best_program = best_programs[best_ind_idx]
#     losses_acc = Vector{Float64}(undef, n)
#     losses_error = Vector{Float64}(undef, n)
#     losses_nll = Vector{Float64}(undef, n)
#     losses_loss = Vector{Float64}(undef, n)
#     indices = collect(1:n)
#     nt = Threads.nthreads()
#     BATCH_SIZE = ceil(Int, n / nt)
#     @info "Running acc_callback"
#     for ith_x in Iterators.partition(indices, BATCH_SIZE)
#         # unpack input nodes
#         # if isdefined(Main, :Infiltrator)
#         # Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
#         # end
#         if use_batch
#             subBatch = batch[ith_x]
#             xs = [i[1] for i in subBatch]
#             ys = [i[2] for i in subBatch]
#         else
#             xs, ys = a.X_test[ith_x], a.Y_test[ith_x]
#         end
#         slot_acc = @view losses_acc[ith_x] # where the thread will write the results
#         slot_error = @view losses_error[ith_x] # where the thread will write the results
#         slot_nll = @view losses_nll[ith_x] # where the thread will write the results
#         slot_loss = @view losses_loss[ith_x] # where the thread will write the results
#         let batch_x = xs
#             batch_y = ys
#             res = @dspawn begin
#                 @W slot_acc
#                 @W slot_error
#                 @W slot_nll
#                 @W slot_loss
#                 t = evaluate_batch_single_individual(batch_x, batch_y, ith_x,
#                     slot_acc, slot_error, slot_nll, slot_loss,
#                     best_program, model_arch, meta_library)
#             end
#         end
#     end
#     final_task = @dspawn @R(losses_acc) @R(losses_error) @R(losses_nll) @R(losses_loss) label = "result"
#     fetch(final_task)
#     final_acc = count(x -> x == 1.0, losses_acc) / length(losses_acc)
#     final_error = count(x -> x == 1.0, losses_error) / length(losses_error)
#     final_nll = mean(losses_nll)
#     final_loss = mean(losses_loss)
#     println("$(a.subset) Accuracy : $(round(final_acc,digits = 2))")
#     println("$(a.subset) Error : $(round(final_error,digits = 2))")
#     println("$(a.subset) Nll: $(round(final_nll,digits = 2))")
#     println("$(a.subset) Loss : $(round(final_loss,digits = 2))")
#     losses_acc, losses_error, losses_nll, losses_error, final_acc, final_error, final_nll, final_loss
# end

# function (a::acc_callback)(program::IndividualPrograms)
#     losses_acc, losses_error, losses_nll, losses_error, final_acc, final_error, final_nll, final_loss = a(
#         [1.0], # ind_performances,
#         Population([]), # population,
#         1, #iteration,
#         run_conf, # run_config,
#         model_arch, # model_architecture,
#         node_config, #node_config,
#         ml, # meta_library,
#         shared_inputs,
#         UTCGP.PopulationPrograms(UTCGP.IndividualPrograms[]), #population_programs,
#         [1.0], #best_losses,
#         [program], # best_programs
#         [1], # elite_idx
#         [],
#     )
#     final_acc
# end

# function evaluate_batch_single_individual(
#     xs::Vector,
#     ys::Vector,
#     indices::SubArray{Int},
#     tracker_acc::SubArray,
#     tracker_error::SubArray,
#     tracker_nll::SubArray,
#     tracker_loss::SubArray, non_shared_programs::IndividualPrograms,
#     model_arch::modelArchitecture,
#     meta_library::MetaLibrary)
#     @debug "Started eval at Thread $(Threads.threadid())"
#     non_shared_programs = deepcopy(non_shared_programs)
#     in_types = model_arch.inputs_types_idx
#     fitnesses_acc = Float64[]
#     fitnesses_error = Float64[]
#     fitnesses_nll = Float64[]
#     fitnesses_loss = Float64[]
#     # append input nodes to pop
#     for (x, y) in zip(xs, ys)
#         UTCGP.reset_programs!(non_shared_programs)
#         input_nodes = [
#             InputNode(value, pos, pos, in_types[pos]) for
#             (pos, value) in enumerate(x)
#         ]
#         replace_shared_inputs!(non_shared_programs, input_nodes)
#         outputs = UTCGP.evaluate_individual_programs(
#             non_shared_programs,
#             model_arch.chromosomes_types,
#             meta_library,
#         )
#         # acc
#         gt = y[1]
#         acc = argmax(outputs) == gt ? 1.0 : 0.0
#         push!(fitnesses_acc, acc)

#         # nll
#         ind_predictions = replace(outputs, NaN => 0.0)
#         pred = argmax(ind_predictions)
#         one_hot_gt = collect(1:2) .== gt
#         nll_loss = Flux.logitcrossentropy(
#             ind_predictions,
#             one_hot_gt,
#             dims=1)
#         nll_loss = clamp(nll_loss, 0.0, 1.0)

#         # MSE
#         ps = softmax(ind_predictions)
#         se = sum((ps .- one_hot_gt) .^ 2)
#         nll_loss = se

#         error_ = pred == gt ? 0.0 : 1.0
#         loss = ACC_WEIGHT * error_ + (1 - ACC_WEIGHT) * nll_loss
#         push!(fitnesses_error, error_)
#         push!(fitnesses_nll, nll_loss)
#         push!(fitnesses_loss, loss)
#     end
#     @debug "Ended eval at Thread $(Threads.threadid())"
#     tracker_acc[1:length(indices)] = fitnesses_acc
#     tracker_error[1:length(indices)] = fitnesses_error
#     tracker_nll[1:length(indices)] = fitnesses_nll
#     tracker_loss[1:length(indices)] = fitnesses_loss
# end

# ###########
# # METRICS #
# ###########

# mutable struct jsonTrackerGA <: AbstractCallable
#     tracker::jsonTracker
#     acc_callback::acc_callback
#     label::String
#     test_losses::Union{Nothing,Vector}
#     best_ind::Union{UTGenome,Nothing}
#     best_loss::Float64
# end
# folder = "metrics_ga"
# f = open(home * "/$folder/" * string(hash) * ".json", "a", lock=true)
# metric_tracker = jsonTracker(h_params, f)
# train_tracker = jsonTrackerGA(metric_tracker,
#     acc_callback([], [], "train"), "Train", nothing, nothing, 0.0) # will use the batch from the training loop to calc metrics
# val_tracker = jsonTrackerGA(metric_tracker,
#     acc_callback(valx, valy, "Val"), "Val", [], nothing, 0.0)

# # TRACK NLL & ACCURACY  => ReRuns for one individual
# function (jtga::jsonTrackerGA)(
#     ind_performances,
#     population,
#     iteration,
#     run_config,
#     model_architecture,
#     node_config,
#     meta_library,
#     shared_inputs,
#     population_programs,
#     best_losses,
#     best_programs,
#     elite_idx,
#     batch
# )
#     losses_acc, losses_error, losses_nll, losses_error, final_acc, final_error, final_nll, final_loss = jtga.acc_callback(ind_performances,
#         population,
#         iteration,
#         run_config,
#         model_architecture,
#         node_config,
#         meta_library,
#         shared_inputs,
#         population_programs,
#         best_losses,
#         best_programs,
#         elite_idx,
#         batch)
#     @warn "JTT $(jtga.label) Fitness : $final_acc"
#     s = Dict("data" => jtga.label, "iteration" => iteration,
#         "accuracy" => final_acc, "error" => final_error, "nll" => final_nll, "loss" => final_loss)
#     if !isnothing(jtga.test_losses)
#         push!(jtga.test_losses, final_acc)
#         if final_acc >= jtga.best_loss
#             # we have a new best
#             jtga.best_ind = deepcopy(population[argmin(best_losses)])
#         end
#         jtga.best_loss = maximum(jtga.test_losses)
#     end
#     write(jtga.tracker.file, JSON.json(s), "\n")
#     flush(jtga.tracker.file)
# end

# ##########
# #   FIT  #
# ##########

# budget_stop = UTCGP.eval_budget_early_stop(Parsed_args["budget"])
# @info "Going To run fit with budget $(Parsed_args["budget"])"
# # best_genome, best_programs, gen_tracker = MAGE_RADIOMICS.PythonCall.GIL.unlock() do

best_genome, best_programs, gen_tracker = UTCGP.fit_ga_meanbatch_mt(
    # @enter fit_ga(
    TRAINDataloader,
    nothing,
    N_REPETITIONS,
    shared_inputs,
    ut_genome,
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
    (train_tracker, val_tracker), #[metric_tracker, test_tracker, sn_writer_callback],
    # Final callbacks ?
    (budget_stop,), #(:default_early_stop_callback,), # 
    nothing #repeat_metric_tracker # .. 
)
# end

# book 
# function save_payload(best_genome,
#     best_programs, gen_tracker,
#     shared_inputs, ml, run_conf,
#     node_config,
#     name::String="best_genome_$hash.pickle")
#     payload = Dict()
#     payload["best_genome"] = deepcopy(best_genome)
#     payload["best_program"] = deepcopy(best_programs)
#     payload["gen_tracker"] = deepcopy(gen_tracker)
#     payload["shared_inputs"] = deepcopy(shared_inputs)
#     payload["ml"] = deepcopy(ml)
#     payload["run_conf"] = deepcopy(run_conf)
#     payload["node_config"] = deepcopy(node_config)
#     payload["best_ind"] = deepcopy(val_tracker.best_ind)
#     payload["best_loss"] = deepcopy(val_tracker.best_loss)

#     genome_path = joinpath(folder, name)
#     open(genome_path, "w") do io
#         @info "Writing payload to $genome_path"
#         write(io, UTCGP.general_serializer(deepcopy(payload)))
#     end
# end

# save_payload(best_genome, best_programs,
#     gen_tracker, shared_inputs,
#     ml, run_conf,
#     node_config)

# save_json_tracker(metric_tracker)
# close(metric_tracker.file)
# en = time()
# println("CUR BUDGET", budget_stop.cur_budget)
# println("TOTAL TIME IN SECONDS:$(en-st)")
# println(maximum(val_tracker.test_losses) * -1)

# exit(0)
# MAGE_RADIOMICS.PythonCall.GIL.unlock() do
#     val_acc = acc_callback(valx, valy, "Val")
#     test_acc = acc_callback(testx, testy, "Test")

#     last_f = Float64[]
#     for final_candidate_prog in best_programs
#         acc = val_acc(final_candidate_prog)
#         push!(last_f, acc)
#     end

#     best_on_val = argmax(last_f)
#     test_accuracy = test_acc(best_programs[best_on_val])

#     # BEST HALF ENSEMBLE 
#     en = time()
#     println(budget_stop.cur_budget)
#     println("TOTAL TIME IN SECONDS:$(en-st)")



#     # LAST FOR IRACE
#     println(string(test_accuracy * -1))
# end
# # sn._execute_command(con, "CHECKPOINT")

# # nodes = sn._execute_command(con, "select count(*) from nodes")
# # edges = sn._execute_command(con, "select count(*) from edges")
# # @show "n nodes $nodes"
# # @show "n edges $edges"

# # DBInterface.close!(con)
# # close(con)
