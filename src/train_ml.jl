using ArgParse
using MLJ
using JLD2
using Statistics
using StatsBase: sample
using DataFrames
using MLJ
DecisionTreeClassifier = MLJ.@load DecisionTreeClassifier pkg = DecisionTree
RandomForestClassifier = MLJ.@load RandomForestClassifier pkg = DecisionTree

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
    "--output_dir"
    arg_type = String
    "--boost_round"
    arg_type = Int
end
rootdir = "./"
Parsed_args = parse_args(s)
@show Parsed_args

BOOST_ROUND = Parsed_args["boost_round"]

include(joinpath(home, "utils", "datasets.jl"))

# READ DATA
data_path = Parsed_args["data_location"]
val_path = Parsed_args["val_data_location"]
has_val_path = val_path != ""
@info "Reading data at $data_path."
@info "Reading val data at $val_path."
data = JLD2.load(data_path)["single_stored_object"]


if !has_val_path
    @info "Since no val data, making own val split at 0.8"
    (trainx, trainy), (valx, valy) = MLUtils.splitobs((data.ys, data.gt); at = 0.8, shuffle = true, stratified = data.gt) # ys is mage preds, gt is true labels from dataset
else
    @info "Loading val data from $val_path."
    val_data = JLD2.load(val_path)["single_stored_object"]
    trainx, trainy = data.ys, data.gt
    valx, valy = val_data.ys, val_data.gt
end

@info "TRAIN SHAPE : X $(size(trainx)) Y $(size(trainy))"
@info "VAL SHAPE : X $(size(valx)) Y $(size(valy))"

train_mat = DataFrame(reduce(vcat, trainx'), :auto)
val_mat = DataFrame(reduce(vcat, valx'), :auto)

RF_METRICS = []
for mx in 1:30
    X = train_mat
    Y = trainy
    model = RandomForestClassifier(n_trees = 200, max_depth = mx, min_samples_split = 10, sampling_fraction = 0.5)
    mach = machine(model, X, categorical(Y)) |> MLJ.fit!

    # ACC
    y_hat = predict_mode(mach, X)
    train_acc = StatisticalMeasures.accuracy(y_hat, Y)
    val_y_hat = predict_mode(mach, val_mat)
    val_acc = StatisticalMeasures.accuracy(val_y_hat, valy)
    @info "Max depth $mx. Train acc : $train_acc. Val acc : $val_acc"
    push!(RF_METRICS, (val_acc = val_acc, train_acc = train_acc, depth = mx, model = deepcopy(model)))
end

sort!(RF_METRICS, by = first, rev = true)


DT_METRICS = []
for mx in 1:20
    X = train_mat
    Y = trainy
    model = DecisionTreeClassifier(max_depth = mx, min_samples_split = 10)
    mach = machine(model, X, categorical(Y)) |> MLJ.fit!

    # ACC
    y_hat = predict_mode(mach, X)
    train_acc = StatisticalMeasures.accuracy(y_hat, Y)
    val_y_hat = predict_mode(mach, val_mat)
    val_acc = StatisticalMeasures.accuracy(val_y_hat, valy)
    @info "Max depth $mx. Train acc : $train_acc. Val acc : $val_acc"
    push!(DT_METRICS, (val_acc = val_acc, train_acc = train_acc, depth = mx, model = deepcopy(model)))
end

sort!(DT_METRICS, by = first, rev = true)


# SAVE = ml_models
folder = Parsed_args["output_dir"]
folder = joinpath(folder, Parsed_args["trial_id"], "ml_models")
isdir(folder) || mkdir(folder)
save_object(
    joinpath(folder, "best_rf_boost_$(BOOST_ROUND).jld2"),
    RF_METRICS[1]
)
save_object(
    joinpath(folder, "best_dt_boost_$(BOOST_ROUND).jld2"),
    DT_METRICS[1]
)

open(joinpath(folder, "best_metrics_boost_$(BOOST_ROUND).txt"), "w") do f
    write(f, RF_METRICS[1].val_acc |> string, "\n")
    write(f, DT_METRICS[1].val_acc |> string)
end
