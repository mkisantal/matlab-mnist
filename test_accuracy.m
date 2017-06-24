%% test_accuracy.m
% Evaluating classification performance on the training set


clear all

% specify the model name!
model_name = 'model.mat';


try load(model_name)
catch
    load model_9896
    disp('model.mat not found, model_9896.mat evaluated instead.')
end
load test_set

%% counting correctly classified images

count=0;

for i=1:size(test_images,2)
    
    input = test_images (:,i);
    
    [hidden_layer_state, output_layer_state] = forward_propagation...
    (input, model.input_to_hidden_weights, model.hidden_to_output_weights,...
     model.hidden_bias, model.output_bias);
 
    [prob, indices] = sort(output_layer_state, 'descend');
    indices = indices-1;
    
    if indices(1) == test_labels(i)
        count = count+1;
    end
end

result = count/size(test_images,2);

fprintf(1, '\n\nCorrectly classified images on test set: %.2f%% \n', result*100);



