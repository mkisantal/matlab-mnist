%% misclassified.m
% Plotting misclassified digits and corresponding output probability
% distributions.

clc
clear all

% specify the model name!
model_name = 'model.mat';


try load(model_name)
catch
    load model_9896
    disp('model.mat not found, model_9896.mat evaluated instead.')
end
load test_set

%% selecting misclassified elements
count=0;
misclass = zeros(1, size(test_images,2));
output = zeros(10, size(test_images,2));
for i=1:size(test_images,2)
    
    input = test_images (:,i);
    
    [hidden_layer_state, output_layer_state] = forward_propagation...
    (input, model.input_to_hidden_weights, model.hidden_to_output_weights,...
     model.hidden_bias, model.output_bias);
 
    [prob, indices] = sort(output_layer_state, 'descend');
    indices = indices-1;
    output(:,i) = output_layer_state;
    if indices(1) ~= test_labels(i)
        count = count+1;
        misclass(i) = 1;
    end
end


missed_images = test_images(:,misclass>0);
missed_output = output(:,misclass>0);
missed_labels = test_labels(:,misclass>0);

  
%% Plotting hidden neuron weights
for i=1:size(missed_images,2)
clc

fprintf(1, 'Number of inorrectly classified images on test set: %.3i \n', count);
fprintf(1,' %i/%i \n\n',count,i)

figure(1)
% misclassified image
subplot(1,2,1)
digit = missed_images(:,i);
kep=reshape(digit,28,28);

% points = linspace(1,28,28);
% [X, Y] = meshgrid(points, points);
% surf(X, Y, kep)
% zlim([-5 5])

%imshow(1-kep)
clims = [-0.3 0.3];
imagesc(1-kep,clims)
colormap(bone)
title('Misclassified digit','interpreter','latex','fontsize',12)
axis equal

% Output probabilities
subplot(1,2,2)
x = 0:1:9;
y = missed_output(:,i);
bar(x,y)
title('Output probabilities','interpreter','latex','fontsize',12)
fprintf(1,'The correct answer was: %i \n\n',missed_labels(i))

disp('Paused')
% print -depsc2 misclass.eps
pause
end

