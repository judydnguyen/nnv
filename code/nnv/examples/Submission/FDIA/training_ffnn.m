%% Training of FFNN for FDIA Classification

t = tic; % track total time for training

%% Load and Clean Data
df = readtable('fdia_data/data1.csv');

% Change column "marker" to numerical value
df.marker = categorical(df.marker);
df.marker = double(df.marker);

% Data cleaning
nanRows = any(ismissing(df), 2);
infRows = any(isinf(table2array(df)), 2);
badRows = nanRows | infRows;
df = df(~badRows, :);

% Display first few rows of cleaned data
% head(df)

%% Split Data
% Separate features and target
X = df(:, df.Properties.VariableNames ~= "marker");
Y = df.marker;

% Get the total number of observations
numObservations = size(df, 1);

% Calculate the number of observations for training (80% for training)
numTrain = floor(0.8 * numObservations);

% Generate a random permutation of indices of observations
randIndices = randperm(numObservations);

% Select indices for training and testing
trainIndices = randIndices(1:numTrain);
testIndices = randIndices(numTrain+1:end);

% Split the data into training and testing sets
XTrain = X(trainIndices, :);
YTrain = Y(trainIndices, :);
XTest = X(testIndices, :);
YTest = Y(testIndices, :);

% Edit input types to feed into CNN else input argument errors
YTrain = categorical(YTrain);
YTest = categorical(YTest);
XTrain = table2array(XTrain);
XTest = table2array(XTest);

%% Training FFNN

inputSize = size(XTrain, 2);
numClasses = numel(categories(YTrain)); 

layers = [ 
    featureInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')

    fullyConnectedLayer(100, 'Name', 'fc1') 
    reluLayer('Name', 'relu1')
    
    fullyConnectedLayer(50, 'Name', 'fc2') 
    reluLayer('Name', 'relu2')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];


options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize', 128, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest,YTest}, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');
 
net = trainNetwork(XTrain,YTrain,layers,options);

% Get Accuracy 
YPred = predict(net, XTest);
YPred = round(YPred);

% Convert YPred to categorical for comparison
YPredCategorical = categorical(YPred);

accuracy = sum(YPredCategorical == YTest) / numel(YTest);
disp ("Validation accuracy = "+string(accuracy));

% Save model
disp("Saving model...");
save('fdia_model_ffnn.mat', 'net', 'accuracy');

toc(t);