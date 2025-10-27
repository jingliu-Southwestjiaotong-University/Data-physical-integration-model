%% Clear Environment
clear all
clc

% Initialize variable to store results
results = table();
numIterations = 5;

%% Data Preparation
data = xlsread('TRAIN.xlsx');
Input = data(:,2:13); % Feature attributes
Output = data(:,1); % Type (2 categories)
[NN mm] = size(data);

%% Bayesian Optimization Objective Function
optFun = @(x) trainAndEvaluateCNN(x, Input, Output, NN);

% Define the range for hyperparameters to be optimized
params = [
    optimizableVariable('convLayer', [16, 64], 'Type', 'integer') % Number of convolutional layers
    optimizableVariable('kernelSize', [2, 5], 'Type', 'integer') % Kernel size
    optimizableVariable('learningRate', [1e-4, 1e-2], 'Transform', 'log') % Learning rate
    optimizableVariable('batchSize', [64, 256], 'Type', 'integer') % Batch size
];

% Run Bayesian Optimization
results_bayes = bayesopt(optFun, params, 'MaxObjectiveEvaluations', 20, 'IsObjectiveDeterministic', false);

% Output the best hyperparameters
disp('Best Parameters from Bayesian Optimization:');
disp(results_bayes.XAtMinObjective);
disp('Best AUC from Bayesian Optimization:');
disp(results_bayes.MinObjective);

%% Bayesian Optimization Objective Function: Train and Evaluate CNN Model
function auc_Test = trainAndEvaluateCNN(x, Input, Output, NN)
    numIterations = 5;
    auc_Test = 0;  % Initial value is 0

    for i = 1:numIterations
        % Randomly generate training and testing sets
        Number = (randperm(size(Input, 1)))'; 
        nn = fix(NN * 0.7);
        Input_train = Input(Number(1:nn),:); 
        Output_train = Output(Number(1:nn),:); 
        Input_test = Input(Number(nn:end),:); 
        Output_test = Output(Number(nn:end),:);
        Input_train = Input_train(:,[1:12]);
        Input_test = Input_test(:,[1:12]);

        % Divide into training and testing sets
        P_train = Input_train';
        T_train = Output_train';
        M = size(P_train, 2);
        P_test = Input_test';
        T_test = Output_test';
        N = size(P_test, 2);

        % Data normalization
        [P_train, ps_input] = mapminmax(P_train, 0, 1);
        P_test  = mapminmax('apply', P_test, ps_input);
        t_train = categorical(T_train)';
        t_test  = categorical(T_test)';

        % Reshape data
        p_train = double(reshape(P_train, 12, 1, 1, M));
        p_test  = double(reshape(P_test , 12, 1, 1, N));

        % Construct network architecture
        layers = [
            imageInputLayer([12, 1, 1])
            convolution2dLayer([x.kernelSize, 1], x.convLayer) % Use Bayesian optimized convolutional layers and kernel size
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer([2, 1], 'Stride', 1)
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer
        ];

        % Set training options
        options = trainingOptions('adam', ...
            'MiniBatchSize', x.batchSize, ...  % Use Bayesian optimized batch size
            'MaxEpochs', 50, ...
            'InitialLearnRate', x.learningRate, ... % Use Bayesian optimized learning rate
            'L2Regularization', 1e-03, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.5, ...
            'LearnRateDropPeriod', 80, ...
            'Shuffle', 'every-epoch', ...
            'ValidationPatience', Inf, ...
            'Verbose', false);

        % Train the model
        net = trainNetwork(p_train, t_train, layers, options);

        % Predict and calculate AUC
        t_sim2 = predict(net, p_test);
        Prob = t_sim2(:,1);
        [P, index] = sort(Prob);
        Outputtest = Output_test(index);
        Pos_num2 = sum(Output_test == 1);
        Neg_num2 = sum(Output_test == 0);
        m_step = 1 / Neg_num2;
        n_step = 1 / Pos_num2;
        m = 1;
        n = 1;
        for i = 1:length(Outputtest)
            if Outputtest(i) == 1
                n = n - n_step;
            else
                m = m - m_step;
            end
            M(i) = m;
            N(i) = n;
        end
        auc_Test = -trapz(N, M); % Calculate AUC
    end
end
