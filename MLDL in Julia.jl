# Install Flux package
using Pkg
Pkg.add("Flux")

Pkg.add("DataFrames")

Pkg.add("CSV")

Pkg.add("MLDatasets")

using Flux
using Flux: onehotbatch, crossentropy, throttle
using DataFrames
using CSV
using MLDatasets

# Load the Iris dataset
df = CSV.File("Iris.csv",header=true) |> DataFrame

# Convert the species column to integers
df.Species = map(x -> x == "Iris-setosa" ? 1 : x == "Iris-versicolor" ? 2 : 3, df.Species)

df

X = Matrix(df[:, 2:5])
X

Y = onehotbatch(convert(Array, df[:, 6]), 1:3)
Y

using Random
Random.seed!(1234)
idxs = randperm(size(X, 1))
train_idxs = idxs[1:round(Int, 0.8 * length(idxs))]
test_idxs = idxs[round(Int, 0.8 * length(idxs)) + 1:end]

train_x = X[train_idxs, :]
train_y = Y[:, train_idxs]
test_x = X[test_idxs, :]
test_y = Y[:, test_idxs]


train_x

train_y

test_x

test_y

print(size(train_x))
print(size(train_y))
print(size(test_x))
print(size(test_y))

using Flux

# Define the neural network
model = Chain(
  Dense(4, 16, relu),
  Dense(16, 8, relu),
  Dense(8, 3),
  softmax
)

# Define the loss function and the optimizer
loss(x, y) = crossentropy(model(x), y)
opt = ADAM(0.1)

# Train the model
n_epochs = 1000
train_data = [(train_x', train_y)]
for i in 1:n_epochs
  Flux.train!(loss, Flux.params(model), train_data, opt)
  if i % 10 == 0
    @show(loss(train_x', train_y))
  end
end

using Pkg
Pkg.add("OneHotArrays")

using OneHotArrays
using Statistics
# Evaluate the performance on the test set
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
@show accuracy(test_x', test_y)
