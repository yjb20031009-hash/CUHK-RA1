% Assuming you have MATLAB's Deep Learning Toolbox installed


% Reshape your input data to fit the network
[gcash_mesh, ghouse_mesh, t_mesh] = meshgrid(gcash,ghouse,[1:tn]);
inputData = [gcash_mesh(:), ghouse_mesh(:), t_mesh(:)]';


% Reshape your output data to fit the network
outputData = [A(:), H(:), C(:) , A1(:), H1(:), C1(:)]';


% Define the layers of your network
layers = [ ...
    featureInputLayer(3) % Input layer for feature vectors of length 5
    fullyConnectedLayer(100) % Fully connected layer with 100 neurons
    reluLayer % Activation function
    fullyConnectedLayer(6) % Fully connected layer with 1 output
    regressionLayer]; % Regression layer for continuous output


% Set the training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',500, ...
    'MiniBatchSize',128, ...
    'InitialLearnRate',0.001, ...
    'ExecutionEnvironment', 'gpu', ... % Use GPU for training
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(inputData, outputData, layers, options);

predictedOutputData = predict(net, inputData);

predictedA = reshape(predictedOutputData(:, 1), size(A));
predictedH = reshape(predictedOutputData(:, 2), size(H));
predictedC = reshape(predictedOutputData(:, 3), size(C));

subplot(3,2,1)
mesh(A(:,:,20))
subplot(3,2,2)
mesh(predictedA(:,:,20))

subplot(3,2,3)
mesh(H(:,:,20))
subplot(3,2,4)
mesh(predictedH(:,:,20))

subplot(3,2,5)
mesh(C(:,:,20))
subplot(3,2,6)
mesh(predictedC(:,:,20))




