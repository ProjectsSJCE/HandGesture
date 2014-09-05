#Set the sizes on the image and frames
frame_size = 100;
base_a = "simpleA/A"; #1918
base_b = "simpleB/B"; #2935
#K = 4900;

% SVM Parameters
#testA = 1918;
C = 0.1; 
sigma = 0.1;
save_file = "nn_params_simple.mat";
#NN parameters
input_layer_size  = 10000;  % 100x100 Input Images of Digits
hidden_layer_size = 200;   % 200 hidden units
num_labels = 2; 
#lambda = 0.5; 98.4%
#lambda = 0.1; #98.6%
lambda_array = [0.1;0.5;1];
max_accuracy = 0;

training_size = 2000;
test_size = 500;#for now test is cross-validation
total = frame_size * frame_size;
X = zeros(training_size, total);
y = zeros(training_size,1);

#for i = 1:2:(training_size),
    y(1:(training_size/2)) = y(1:(training_size/2)) + 1;
#    endfor;

y_nn = y + 1; #mapping 1 0 of svm to 2 1 in nn

fprint("Starting");

#y = rand(training_size,1);
#y(y<0.5) = 0;
#y(y>=0.5) = 1;
#y(1:end) = 0;
#Training
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

###Testing
#load(save_file);
X_test = zeros(test_size, total);
#base_a = "A";
#base_b = "B";
for iter = ((training_size/2)+1):((training_size/2)+(test_size/2)),
#Reads the Image
#        key = iter;
        key = iter - (training_size/2);
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
#        key = iter;
        key = iter - (training_size/2);
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

#model = svmTrain(X, y, C, @gaussianKernel);
#model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
#model = svmTrain(X, y, C, @linearKernel);
#model = svmTrain(X, y, C, @linearKernel, 1e-3, 50);
#model = svmTrain(X, y, C, @linearKernel);
#model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

result = ones(test_size,1);
result(1:(test_size/2)) = result(1:(test_size/2)) + 1;
lambda = 0;

for lambda_iter=1:3
fprintf("training");
lambda = lambda_array(lambda_iter);
#Neural networks
nn_params = train_nn(input_layer_size,hidden_layer_size,num_labels,lambda,X,y_nn);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
disp("Done with nn training, press enter");

##model = svmTrain(X, y, C, @linearKernel, 0.1, 0.1);


#disp("predicting -SVM");
##SVM prediction
#predictions_svm = svmPredict(model, X_test);
#predictions_svm += 1;
#accuracy = sum(result==predictions_svm) / test_size * 100;
##disp(predictions_svm);
#disp(accuracy);

#NN prediction
disp("predicting -NN");
predictions_nn = predict(Theta1, Theta2, X_test);
accuracy = (sum((result==predictions_nn)) / test_size) * 100;
#disp(predictions_nn);
fprintf("accuracy = ");
disp(accuracy);
fprintf("lambda = ");
disp(lambda);

#Saving nn paramters in save_file
if (accuracy > max_accuracy)
    fprintf("Saving neural network parameters.");
    junk = save_nn_parameters(Theta1, Theta2, save_file);
    accuracy = max_accuracy;
else
    junk = 0;
    endif;
endfor;
##===========================================================================================================
