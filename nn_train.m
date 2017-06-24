%% Training a neural network from scratch on MNIST
clc
clear all
start_time = clock;
%memory
%% Hyperparameters

% SET HYPERPARAMETERS HERE.
batchsize = 200;    % Mini-batch size.
learning_rate0 = 0.1;
momentum = 0.86;    % Momentum
numhid = 32;        % hidden layer size
epochs = 5;
lambda = 5;         % L2 regularization
dropout = 0;        % activating dropout


%% Loading MNIST data

[train_images, train_labels, valid_images, valid_labels,...
    test_images, test_labels] = load_images(batchsize);

[input_size, batchsize, numbatches]=size(train_images);
output_size = 10;

%% INITIALIZATION
% new weight initialization
input_to_hidden_weights  = 1/sqrt(input_size) * randn(input_size, numhid);
hidden_to_output_weights =  zeros(numhid, output_size);

hidden_bias = zeros(numhid,1);
output_bias = zeros(output_size,1);

CE_array = zeros(numbatches,epochs);

% initializing gradient matrices
input_to_hidden_weights_delta = zeros(input_size, numhid);
hidden_to_output_weights_delta = zeros(numhid, output_size);

hidden_bias_delta = zeros(numhid,1);
output_bias_delta = zeros(output_size,1);


% other
tiny = exp(-30);                        % avoiding log(0)
show_training_CE_after = 100;           % for tracking progress
show_validation_CE_after = 100;         % for tracking progress
count = 0;

CE_array_avg = zeros(floor(numbatches/show_training_CE_after),epochs);
CE_array_valid = zeros(floor(numbatches/show_validation_CE_after),epochs);

%% LOOP OVER EPOCHS
for epoch = 1:epochs;
  
  fprintf(1, 'Epoch %d\n', epoch);
  this_chunk_CE = 0;
  trainset_CE = 0;
  
%%  Scheduling learning rate
learning_rate = learning_rate0 / (1+((epoch-1)/10)^2);
weight_dec = 1 - learning_rate * lambda / (batchsize * numbatches);
disp(' Learning rate ')
disp(learning_rate)


for m=1:numbatches      %loop over mini-batches
%% Dropout
if dropout == 1
    
    % storing complete matrices, scaling up!
    full_input_to_hidden_weights  = 2 * input_to_hidden_weights;
    full_hidden_to_output_weights = 2 * hidden_to_output_weights;
    full_hidden_bias              = 2 * hidden_bias;
    
    full_input_to_hidden_weights_delta  = 2 * input_to_hidden_weights_delta;
    full_hidden_to_output_weights_delta = 2 * hidden_to_output_weights_delta;
    full_hidden_bias_delta              = 2 * hidden_bias_delta;
    
    % random element selection
    selecting_vector = sort(randperm(numhid,floor(numhid/2)));
    
    input_to_hidden_weights  = full_input_to_hidden_weights( :, selecting_vector) ;
    hidden_to_output_weights = full_hidden_to_output_weights( selecting_vector, :);
    hidden_bias              = full_hidden_bias( selecting_vector, :);
    
    input_to_hidden_weights_delta  = full_input_to_hidden_weights_delta(:, selecting_vector);
    hidden_to_output_weights_delta = full_hidden_to_output_weights_delta(selecting_vector, :);
    hidden_bias_delta              = full_hidden_bias_delta(selecting_vector, :);
end


%% Forward propagate
input_batch = train_images(:,:,m);

[hidden_layer_state, output_layer_state] = forward_propagation...
    (input_batch, input_to_hidden_weights, hidden_to_output_weights,...
     hidden_bias, output_bias);
 
 
%% Error fcn

expansion_matrix = eye(output_size);
target_batch     = train_labels(:,:,m);
target_vectors   = expansion_matrix(:,target_batch+1); % +1 avoiding zero indices


% LOG Likelihood error function
CE = -sum(sum(target_vectors .* log(output_layer_state + tiny)))/batchsize;
CE_array(m,epoch)=CE;

% show avg error
count =  count + 1;
this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count;
trainset_CE = trainset_CE + (CE - trainset_CE) / m;
% fprintf(1, '\rBatch %d Train CE %.3f', m, this_chunk_CE);
if mod(m, show_training_CE_after) == 0
    fprintf(1, '\n');
    count = 0;
    CE_array_avg(m/show_training_CE_after,epoch) = this_chunk_CE;
    this_chunk_CE = 0;
end




%% Backpropagation

% Error fcn derivative
error_deriv = output_layer_state - target_vectors;

% output layer
hid_to_output_weights_grad =  hidden_layer_state * error_deriv';
output_bias_grad           = sum(error_deriv, 2);

% hidden layer
back_propagated_deriv = (hidden_to_output_weights * error_deriv) ...
      .* hidden_layer_state .* (1 - hidden_layer_state);

input_to_hid_weights_grad = input_batch * back_propagated_deriv';
hidden_bias_grad          = sum(back_propagated_deriv, 2);

%% Updating weights and biases

% input_to_hidden_weights
input_to_hidden_weights_delta = momentum .* input_to_hidden_weights_delta...
    + input_to_hid_weights_grad ./ batchsize;
input_to_hidden_weights = input_to_hidden_weights * weight_dec...
    - learning_rate * input_to_hidden_weights_delta;

% hidden_to_output_weights
hidden_to_output_weights_delta = momentum .* hidden_to_output_weights_delta...
    + hid_to_output_weights_grad ./ batchsize;
hidden_to_output_weights = hidden_to_output_weights * weight_dec...
    - learning_rate * hidden_to_output_weights_delta;

% hidden_bias
hidden_bias_delta = momentum .* hidden_bias_delta...
    + hidden_bias_grad ./ batchsize;
hidden_bias = hidden_bias - learning_rate * hidden_bias_delta;

% output_bias
output_bias_delta = momentum .* output_bias_delta...
    - output_bias_grad ./ batchsize;
output_bias = output_bias - learning_rate * output_bias_delta;

%% Merging weight and bias matrices for dropout
if dropout == 1
% update rows in complete matrices
full_input_to_hidden_weights( :, selecting_vector)  = input_to_hidden_weights;
full_hidden_to_output_weights( selecting_vector, :) = hidden_to_output_weights;
full_hidden_bias( selecting_vector, :)              = hidden_bias;

full_input_to_hidden_weights_delta(:, selecting_vector)  = input_to_hidden_weights_delta;
full_hidden_to_output_weights_delta(selecting_vector, :) = hidden_to_output_weights_delta;
full_hidden_bias_delta(selecting_vector, :)              = hidden_bias_delta;

% switch back to standard matrix names, scaling!
input_to_hidden_weights  = 0.5 * full_input_to_hidden_weights;
hidden_to_output_weights = 0.5 * full_hidden_to_output_weights;
hidden_bias              = 0.5 * full_hidden_bias;

input_to_hidden_weights_delta  =  0.5 * full_input_to_hidden_weights_delta;
hidden_to_output_weights_delta =  0.5 * full_hidden_to_output_weights_delta;
hidden_bias_delta              =  0.5 * full_hidden_bias_delta;

end
    
%% Validation
if mod(m, show_validation_CE_after) == 0
    fprintf(1, '\rRunning validation ...');
    
    target_vectors_validation = expansion_matrix(:,valid_labels+1); % +1 avoiding zero indices
    datasetsize_validation    = size(valid_images, 2);

    [hidden_layer_state, output_layer_state] = forward_propagation...
    (valid_images, input_to_hidden_weights, hidden_to_output_weights,...
     hidden_bias, output_bias);
 
    CE_valid = -sum(sum(target_vectors_validation .* log(output_layer_state + tiny)))...
    /datasetsize_validation;
    
    CE_array_valid(m/show_validation_CE_after,epoch) = CE_valid;
    
    fprintf(1, ' Validation CE %.3f\n', CE_valid);

end
end % end loop over Mini-Batches
fprintf(1, '\rAverage Training CE %.3f\n', trainset_CE);

end % end loop over Epochs

fprintf(1, 'Finished Training.\n');
fprintf(1, 'Final Training CE %.3f\n', trainset_CE);

%% EVALUATE ON VALIDATION SET.
fprintf(1, '\rRunning validation ...');

target_vectors_validation = expansion_matrix(:,valid_labels+1); % +1 avoiding zero indices
datasetsize_validation    = size(valid_images, 2);

[hidden_layer_state, output_layer_state] = forward_propagation...
    (valid_images, input_to_hidden_weights, hidden_to_output_weights,...
     hidden_bias, output_bias);
 
CE = -sum(sum(target_vectors_validation .* log(output_layer_state + tiny)))...
    /datasetsize_validation;
fprintf(1, '\rFinal Validation CE %.3f\n', CE);


%% EVALUATE ON TEST SET.
fprintf(1, '\rRunning test ...');

target_vectors_validation = expansion_matrix(:,test_labels+1); % +1 avoiding zero indices
datasetsize_test          = size(test_images, 2);

[hidden_layer_state, output_layer_state] = forward_propagation...
    (test_images, input_to_hidden_weights, hidden_to_output_weights,...
     hidden_bias, output_bias);
 
CE = -sum(sum(target_vectors_validation .* log(output_layer_state + tiny)))...
    /datasetsize_test;
fprintf(1, '\rFinal Test CE %.3f\n', CE);

%%
end_time = clock;
diff = etime(end_time, start_time);
fprintf(1, 'Training took %.2f seconds\n', diff);

%% SAVE model

model.input_to_hidden_weights  = input_to_hidden_weights;
model.hidden_to_output_weights = hidden_to_output_weights;
model.hidden_bias              = hidden_bias;
model.output_bias              = output_bias;

save model model

plot_CE(CE_array,CE_array_avg,CE_array_valid)
test_accuracy




