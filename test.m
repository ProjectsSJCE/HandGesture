#Set the sizes on the image and frames
max_row = 1000;
max_col = 1000;
frame_size = 40;
flag = 0;
base_a = "A";
base_b = "B";
K = 400;

C = 0.05;# The value with least validation error
#sigma = 

training_size = 8;
test_size = 6;
total = frame_size * frame_size;
#repeat = 1;
X = zeros(training_size, total);
y = zeros(training_size,1);
for i = 1:2:(training_size),
    y(i) = 1;
    endfor;
y_nn = y + 1;
flag = 0;
#Training
for iter = 1:training_size,
#Reads the Image
    if flag == 0
        key = (iter + 1) / 2;
        key = num2str(key);
        file = strcat(base_a, key, ".jpeg");
        flag = 1;
    else
        key = iter / 2;
        key = num2str(key);
        file = strcat(base_b, key, ".jpeg");
        flag = 0;
    endif;

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
disp("Starting.");
#Does featureNormalize
[img_norm, mu, sigma] = featureNormalize(X);
#imshow(img_norm); #Undocument to see the normalised hand
#pause;

disp("Now starting pca. Press enter.");
pause;
[U, S] = pca(img_norm, frame_size, frame_size);
disp("Done with pca.");
Z = projectData(img_norm, U, K);
X = Z;
#imshow(Z); #Undocument to see pca
disp("paused, press enter");
pause;

disp("training");
#model = svmTrain(X, y, C, @gaussianKernel);
#model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
model = svmTrain(X, y, C, @linearKernel);

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 20;   % 25 hidden units
num_labels = 2; 
lambda = 1;

nn_params = train_nn(input_layer_size,hidden_layer_size,num_labels,lambda,X,y_nn);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
disp("Done with nn training, press enter");
#model = svmTrain(X, y, C, @linearKernel, 0.1, 0.1);
#Testing

flag = 0;
X_test = zeros(test_size, total);

for iter = (training_size+1):(training_size+test_size),
#Reads the Image
    if flag == 0
        key = (iter + 1) / 2;
        key = num2str(key);
        file = strcat(base_a, key, ".jpeg");
        flag = 1;
    else
        key = iter / 2;
        key = num2str(key);
        file = strcat(base_b, key, ".jpeg");
        flag = 0;
    endif;
    img = imread(file);
    [img] = blacken_palm(img, frame_size);
    #imshow(img);
    #pause;
    img_transpose = img';
    for index = 1:total
        X_test(iter-training_size,index) = img_transpose(index);
    endfor;
endfor;

#Does featureNormalize
[img_norm, mu, sigma] = featureNormalize(X_test);
#imshow(img_norm); #Undocument to see the normalised hand
disp("Paused, Press enter.");
pause;

disp("Now starting pca");
#PCA
[U, S] = pca(img_norm, frame_size, frame_size);
disp("Done with pca");
Z = projectData(img_norm, U, K);
#imshow(Z); #Undocument to see pca
#disp("press enter");
#pause;
X_test = Z;

disp("predicting -SVM");
#SVM prediction
predictions_svm = svmPredict(model, X_test);
predictions_svm += 1;
disp(predictions_svm);
#NN prediction
disp("predicting -NN");
predictions_nn = predict(Theta1, Theta2, X_test);
disp(predictions_nn);
##===========================================================================================================
