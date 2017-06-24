function [train_images_batch, train_labels_batch, valid_images, valid_labels, test_images, test_labels] = load_images(N)
%% LOADING MNIST images

train_images2 = loadMNISTImages('train-images.idx3-ubyte');
train_labels2 = loadMNISTLabels('train-labels.idx1-ubyte');

test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte')';


% separating image set for validation
train_images = train_images2(:,1:50000);
train_labels = train_labels2(1:50000,1);

valid_images = train_images2(:,50001:60000);
valid_labels = train_labels2(50001:60000)';


%% -------------------- to use expanded trainset, uncomment this section----------------
% load expanded_trainset500k
% train_labels = repmat(train_labels,1,size(train_images,2)/50000); 
%% Segmenting training set to mini-batches

M = floor(size(train_images,2)/N);

train_images_batch = reshape(train_images(:,1:M*N),size(train_images,1),N,M);
train_labels_batch = reshape(train_labels(1:M*N)',1,N,M);           % labels transposed!


save test_set test_images test_labels
% save imported_imageset train_images valid_images test_images

end


