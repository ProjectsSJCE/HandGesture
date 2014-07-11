#Set the sizes on the image and frames
max_row = 1000;
max_col = 1000;
frame_size = 100;
flag = 0;
base_a = "A";
base_b = "B";
K = 100;
training_size = 8;
test_size = 6;
total = frame_size * K;
X = zeros(training_size*20, total);
y = zeros(160,1);
for i = 1:2:160,
    y(i) = 1;
    end;
for count = 1:20,
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
end;

img = imread (file);
#rgb = img;
#[x, map] = rgb2ind(img);
#img = ind2gray (x, map);
#[row,col] = size(img);
#img = img / 255.0;
[img] = blacken_palm(img, frame_size);
#imshow(img);
#pause;
#Have to decide on C and sigma values
C = 0.05;
#sigma = 0.3;
#putting all features into a row vector
#but does this make sense? After applying PCA in the way we did apply
#disp((training_size*(count-1)) + iter);
for index = 1:total
    X((training_size*(count-1)) + iter,index) = img(index);
    end;
#disp("saar");
end;
end;
#y = [1;0]; 
#X = [1,2;3,4]; 
#C = 0.03;
disp("training");
#size(X)
#model = svmTrain(X, y, C, @gaussianKernel);
#model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
model = svmTrain(X, y, C, @linearKernel);
#model = svmTrain(X, y, C, @linearKernel, 0.1, 0.1);
#Testing
flag = 0;
X_test = zeros(test_size, total);
for iter = (training_size+1):(test_size+training_size),
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
end;

img = imread (file);
#rgb = img;
#[x, map] = rgb2ind(img);
#img = ind2gray (x, map);
#[row,col] = size(img);
#img = img / 255.0;
[img] = blacken_palm(img, frame_size);
#imshow(img);
#pause;
for index = 1:total
    X_test((iter-8),index) = img(index);
    end;
#disp("saar2");
end;
disp("predicting");
predictions = svmPredict(model, X_test);
disp(predictions);
##===========================================================================================================
