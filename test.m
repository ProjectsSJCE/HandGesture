#Set the sizes on the image and frames
frame_size = 100;
base_a = "/media/c19aea56-3b44-4118-a3ec-d9abc9b15801/programming/projects/dataA/A"; #1918
base_b = "/media/c19aea56-3b44-4118-a3ec-d9abc9b15801/programming/projects/dataB/B"; #2935

save_file_nn = "nn_params.mat";
save_file_svm = "svm_params.mat";
#NN parameters
input_layer_size  = 10000;  % 100x100 Input Images of Digits
hidden_layer_size = 200;   % 200 hidden units
num_labels = 2; 
lambda = 1;

lambda_array = [0.1;0.5;1];
max_accuracy = 0;
positive_training_size = 0;
negative_training_size = 500;
test_size = 500;#for now test is cross-validation
total = frame_size * frame_size;

X = zeros((positive_training_size + negative_training_size), total);
y = zeros((positive_training_size + negative_training_size),1);
y(1:positive_training_size) = 1;

y_nn = y + 1; #mapping 1 0 of svm to 2 1 in nn

fprintf("Starting\n");

#Training
for iter = 1:positive_training_size,
#Reads the Image
    key = iter;
    key = num2str(key);
    file = strcat(base_a, key, ".jpg");
    img = imread(file);
    imshow(img);
    pause;

    [img] = blacken_palm(img, frame_size);
    imshow(img);
    pause;
    img_transpose = img';

#putting all features into a row vector
    for index = 1:total,
        X(iter,index) = img_transpose(index);
    endfor;
endfor;

for iter = 44:negative_training_size
#Reads the Image
    key = iter;
    key = num2str(key);
    file = strcat(base_b, key, ".jpg");
    img = imread(file);
    imshow(img);
    pause;
    [img] = blacken_palm(img, frame_size);
    imshow(img);
    pause;
    img_transpose = img';

#putting all features into a row vector
    for index = 1:total,
        X(iter+positive_training_size,index) = img_transpose(index);
    endfor;
endfor;

####Testing
#load(save_file_nn);
#load(save_file_svm);
X_test = zeros(test_size, total);
#base_a = "complexA/A"; #1918
#base_b = "complexB/B"; #2935
training_size = 3000;
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
training_size = 4000;
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

model = svmtrain(X, y);

result = zeros(test_size,1);
result(1:(test_size/2)) = result(1:(test_size/2)) + 1;
fprintf("Done loading images\n");

#for lambda_iter=1:3
#fprintf("training");
#lambda = lambda_array(lambda_iter);
#Neural networks
nn_params = train_nn(input_layer_size,hidden_layer_size,num_labels,lambda,X,y_nn);

#% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
disp("Done with nn training, press enter");

disp("predicting -SVM");
#SVM prediction
predictions_svm = svmclassify(model, X_test);
#predictions_svm += 1;
accuracy = sum(result==predictions_svm) / test_size * 100;
#disp(predictions_svm);
disp(accuracy);
disp("Details of SVM prediction");
tp = sum((result==1).*(predictions_nn==1));
fp = sum((result==0).*(predictions_nn==1));
tn = sum((result==0).*(predictions_nn==0));
fn = sum((result==1).*(predictions_nn==0));
disp(tp);
disp(fp);
disp(tn);
disp(fn);

#NN prediction
result = result + 1;
disp("predicting -NN");
predictions_nn = predict(Theta1, Theta2, X_test);
accuracy = (sum((result==predictions_nn)) / test_size) * 100;
#disp(predictions_nn);
fprintf("accuracy = ");
disp(accuracy);

disp("Details of NN prediction");
tp = sum((result==2).*(predictions_nn==2));
fp = sum((result==1).*(predictions_nn==2));
tn = sum((result==1).*(predictions_nn==1));
fn = sum((result==2).*(predictions_nn==1));
disp(tp);
disp(fp);
disp(tn);
disp(fn);

#Saving nn paramters in save_file
#if (accuracy > max_accuracy)
#    fprintf("Saving neural network parameters.");
#    junk = save_nn_parameters(Theta1, Theta2, save_file_nn);
#    accuracy = max_accuracy;
#else
#    junk = 0;
#    endif;
#endfor;

#Saving svm paramters in save_file
#    fprintf("Saving svm parameters.");
#    junk = save_svm_parameters(model, save_file_svm);
#endfor;

##===========================================================================================================
