%% Robustness verification of FFNN against FDIAs

%% Load data into NNV 

% Specify the ONNX file to load 
% 50_100_50 model -> onnx model 1
onnx_model_1 = fullfile('onnx_models','fdia_model_ffnn_pytorch_50_100_50.onnx');
% 100_200_100 model -> onnx model 2
onnx_model_2 = fullfile('onnx_models','fdia_model_ffnn_pytorch_100_200_100.onnx');
% 200_400_200 model -> onnx model 3
onnx_model_3 = fullfile('onnx_models','fdia_model_ffnn_pytorch_200_400_200.onnx');

% Load the ONNX file as DAGNetwork
% ADJUST which onnx file here
netONNX = importONNXNetwork(onnx_model_3, 'OutputLayerType', 'classification', 'InputDataFormats', {'BC'});

% Convert the DAGNetwork to NNV format
net = matlab2nnv(netONNX);

% Jimmy Rigged Fix: manually edit ouput size
net.OutputSize = 2;

% Load data 
load('fdia_test_data.mat');
X_test_loaded = X;
y_test_loaded = y;

% Normalize features in X_test_loaded
min_values = min(X_test_loaded, [], 1);
max_values = max(X_test_loaded, [], 1);

% Ensure no division by zero for constant features
variableFeatures = max_values - min_values > 0;
min_values(~variableFeatures) = 0; % Avoids changing constant features
max_values(~variableFeatures) = 1; % Avoids division by zero

% Apply normalization
X_test_loaded(:, variableFeatures) = (X_test_loaded(:, variableFeatures) - min_values(variableFeatures)) ./ (max_values(variableFeatures) - min_values(variableFeatures));

% Count total observations
total_obs = size(X_test_loaded, 1);
% disp(['There are total ', num2str(total_obs), ' observations']);

% ADJUST epsilon value here
epsilons = [0.005,0.01,0.025];

%% Main Computation

% to save results (robustness and time)
results = zeros(20,2);

% Define reachability method
reachOptions = struct;
reachOptions.reachMethod = 'exact-star';

% Iterate through the different epsilons
for e=1:length(epsilons)
    % Iterate trhough all observations
    for i=1:100
        observation = X(i, :);
        % Extract the corresponding label for the current observation
        target = y(i);
        target = single(target);
        
        % Adversarial attack
        if epsilons(e) ~= 0
              % Apply epsilon perturbation to variable features
            perturbation = zeros(size(observation));
            perturbation(variableFeatures) = epsilons(e) * (2*rand(1, sum(variableFeatures)) - 1); % Perturbation in [-epsilon, epsilon]
            perturbedObservation = observation + perturbation;
        end
    
        % Ensure the values are within the valid range for pixels ([0 255])
        lb_min = zeros(size(X_test_loaded)); % Lower bound is 0 for all features
        ub_max = ones(size(X_test_loaded));  % Upper bound is 1 for all features
        lb_clip = max(perturbedObservation, lb_min(i, :));
        ub_clip = min(perturbedObservation, ub_max(i, :));
        IS = ImageStar(lb_clip, ub_clip); % this is the input set we will use
        
        % % Let's evaluate the image and the lower and upper bounds to ensure these
        % % are correctly classified
        % 
        % if ~mod(i,50)
        %     disp("Verifying image "+string(i)+" out of "+string(N)+" in the dataset...");
        % end
    
        % Begin tracking time after input set is created
        t = tic;
    
        % Evaluate input image
        Y_outputs = net.evaluate(observation); 
        [~, yPred] = max(Y_outputs);
        
        % Evaluate lower and upper bounds
        LB_outputs = net.evaluate(lb_clip);
        [~, LB_Pred] = max(LB_outputs); 
        UB_outputs = net.evaluate(ub_clip);
        [~, UB_Pred] = max(UB_outputs);
    
        % Check if outputs are violating robustness property
        if any([yPred, LB_Pred, UB_Pred] ~= target)
            results(i,1) = 0;
            results(i,2) = toc(t);
            % if counterexample found, no need to do any reachability analysis
            continue;
        end
        
        % If not, we verify the robustness using reachability analysis
        %  - Use the NN.verify_robustness function
    
        % A common approach would be to use some refinement approach like
        %   - Try first with faster approx method, if not robust, compute the
        %   exact reachability analysis
    
        % For the purpose of this tutorial, we are only going to do the
        % approximate method
    
        % Temporarily turn off all warnings
        % Getting warning - Warning: Changing input set precision to single 
        warning('off','all');
    
        % Verification
        results(i,1) = net.verify_robustness(IS, reachOptions, target);
        results(i,2) = toc(t);
        
        % Restore warning state
        warning('on','all');
    
    
    end
    
    % Get summary results
    N = length(results);
    rob = sum(results(:,1) == 1);
    not_rob = sum(results(:,1) == 0);
    unk = sum(results(:,1) == 2);
    totalTime = sum(results(:,2));
    avgTime = totalTime/N;
    
    
    % Print results to screen
    disp("======= ROBUSTNESS RESULTS ==========")
    disp("Epsilon value: "+string(epsilons(e)))
    disp(" ");
    disp("Number of robust samples = "+string(rob)+ ", equivalent to " + string(100*rob/N) + "% of the samples.");
    disp("Number of not robust samples = " +string(not_rob)+ ", equivalent to " + string(100*not_rob/N) + "% of the samples.")
    disp("Number of unknown samples = "+string(unk)+ ", equivalent to " + string(100*unk/N) + "% of the samples.");
    disp(" ");
    disp("It took a total of "+string(totalTime) + " seconds to compute the verification results, an average of "+string(avgTime)+" seconds per image");
    
    % Save results
    save('results_verify_fdia_ffnn.mat', 'results');
end