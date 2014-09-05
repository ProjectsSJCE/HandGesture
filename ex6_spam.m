%% Initialization
clear ; close all; clc
%% ==================== Part 1: Email Preprocessing ====================
%  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
%  to convert each email into a vector of features. In this part, you will
%  implement the preprocessing steps for each email. You should
%  complete the code in processEmail.m to produce a word indices vector
%  for a given email.



#Set the sizes on the image and frames
max_row = 1000;
max_col = 1000;
frame_size = 100;
flag = 0;
base_a = "dataA/A"; #1918
base_b = "dataB/B"; #2935
#K = 4900;

% SVM Parameters

C = 0.0001; sigma = 0.1;

#NN parameters
input_layer_size  = 10000;  % 100x100 Input Images of Digits
hidden_layer_size = 200;   % 200 hidden units
num_labels = 2; 
lambda = 1;

training_size = 200;
test_size = 30;#for now test is cross-validation
total = frame_size * frame_size;
X = zeros(training_size, total);
y = zeros(training_size,1);

#for i = 1:2:(training_size),
    y(1:(training_size/2)) = y(1:(training_size/2)) + 1;
#    endfor;
#y = rand(training_size,1);
#y(y<0.5) = 0;
#y(y>=0.5) = 1;
for iter = 1:(training_size/2),
#Reads the Image
    key = iter;
    key = num2str(key);
    file = strcat(base_a, key, ".jpg");
    img = imread(file);
    [img] = blacken_palm(img, frame_size);
#    imshow(img);
#    pause;
    img_transpose = img';

#putting all features into a row vector
    for index = 1:total,
#       X((training_size*(count-1)) + iter,index) = img(index);
        X(iter,index) = img_transpose(index);
    endfor;
    
endfor;

for iter = 1:(training_size/2),
#Reads the Image
    key = iter;
    key = num2str(key);
    file = strcat(base_b, key, ".jpg");
    img = imread(file);
    [img] = blacken_palm(img, frame_size);
#    imshow(img);
#    pause;
    img_transpose = img';

#putting all features into a row vector
    for index = 1:total,
#       X((training_size*(count-1)) + iter,index) = img(index);
        X(iter+(training_size/2),index) = img_transpose(index);
    endfor;
    
endfor;

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.001;
model = svmTrain(X, y, C, @linearKernel);
p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);
pause;
%% =================== Part 4: Test Spam Classification ================
%  After training the classifier, we can evaluate it on a test set. We have
%  included a test set in spamTest.mat

% Load the test dataset
% You will have Xtest, ytest in your environment
#load('spamTest.mat');
result = ones(test_size,1);
result(1:(test_size/2)) = result(1:(test_size/2)) + 1;

##Testing

X_test = zeros(test_size, total);
#base_a = "dataA/A";
#base_b = "datab";
for iter = ((training_size/2)+1):((training_size/2)+(test_size/2)),
#Reads the Image
        key = iter;
#        key = iter - (training_size/2);
        key = num2str(key);
        file = strcat(base_a, key, ".jpg");
        flag = 1;
    img = imread(file);
    [img] = blacken_palm(img, frame_size);
#    imshow(img);
#    pause;
    img_transpose = img';
    for index = 1:total
        X_test(iter-(training_size/2),index) = img_transpose(index);
    endfor;
endfor;

for iter = ((training_size/2)+1):((training_size/2)+(test_size/2)),
#Reads the Image
        key = iter;
#        key = iter - (training_size/2);
        key = num2str(key);
        file = strcat(base_b, key, ".jpg");
        flag = 1;
    img = imread(file);
    [img] = blacken_palm(img, frame_size);
#    imshow(img);
#    pause;
    img_transpose = img';
    for index = 1:total
        X_test(iter-(training_size/2)+(test_size/2),index) = img_transpose(index);
    endfor;
endfor;

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')
#[C, sigma] = dataset3Params(X, y, X_test, result);
p = svmPredict(model, X_test);
fprintf('Test Accuracy: %f\n', mean(double(p == result)) * 100);
pause;
