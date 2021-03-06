#Set the sizes on the image and frames
max_row = 1000;
max_col = 1000;
frame_size = 40;
flag = 0;
base_a = "A";
base_b = "B";
K = 20;
training_size = 8;
test_size = 6;

total = frame_size * K;

X = zeros(training_size*100, total);
y = zeros(800,1);
for i = 1:2:800,
    y(i) = 1;
    end;
for count = 1:100,
#Training
for iter = 1:training_size,
#Reads the Image
if flag == 0
    key = (iter + 1) / 2;
    key = num2str(key);
    file = strcat(base_a, key, ".jpeg");
#    file = "palm_white.jpg";
    flag = 1;
else
    key = iter / 2;
    key = num2str(key);
    file = strcat(base_b, key, ".jpeg");
#    file = "blue.jpg";
    flag = 0;
end;

img = imread (file);

#Blackens the palm
[img] = blacken_palm(img, frame_size);
#imshow(img); #undocument to see blackened hand
#pause;

##Filters the image
#frame_size = 80;
#img = crop_hand(img, frame_size);
#imshow(img);

#Applying a high pass filter, since it is used for edge detection, seems to do a better job
#high_pass = [0.25,0.5,0.25;0.5,9,0.5;0.25,0.5,0.25];
#img = imfilter(img, high_pass, "replicate");
#imshow(img); #Undocument to see filtered hand

#Does featureNormalize
[img_norm, mu, sigma] = featureNormalize(img);
#imshow(img_norm); #Undocument to see the normalised hand
#pause;

#disp("Now starting pca");
[U, S] = pca(img_norm, frame_size, frame_size);
#disp("Done with pca");
Z = projectData(img_norm, U, K);
#imshow(Z); #Undocument to see pca

#Have to decide on C and sigma values
#C = 1;
#sigma = 0.1;

#putting all features into a row vector
#but does this make sense? After applying PCA in the way we did apply
for index = 1:total
#    X(iter,index) = Z(index);
    X((training_size*(count-1)) + iter,index) = Z(index);
    end;
end;
end;
sigma = 0.3;
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);
#model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2));
#Testing
flag = 0;
X_test = zeros(test_size, total);
for iter = (training_size+1):(test_size+training_size),
#Reads the Image
if flag == 0
    key = (iter + 1) / 2;
    key = num2str(key);
#    file = strcat(base_b, key, ".jpeg");
    file = "blue.jpg";
    flag = 1;
else
    key = iter / 2;
    key = num2str(key);
#    file = strcat(base_b, key, ".jpeg");
    file = "palm_white.jpg";
    flag = 0;
end;

img = imread (file);

#Blackens the palm
[img] = blacken_palm(img, frame_size);
#imshow(img); #undocument to see blackened hand
#pause;

##Filters the image
#frame_size = 80;
#img = crop_hand(img, frame_size);
#imshow(img);

#Applying a high pass filter, since it is used for edge detection, seems to do a better job
#high_pass = [0.25,0.5,0.25;0.5,9,0.5;0.25,0.5,0.25];
#img = imfilter(img, high_pass, "replicate");
#imshow(img); #Undocument to see filtered hand

#Does featureNormalize
[img_norm, mu, sigma] = featureNormalize(img);
#imshow(img_norm); #Undocument to see the normalised hand
#pause;

#disp("Now starting pca");
[U, S] = pca(img_norm, frame_size, frame_size);
#disp("Done with pca");
Z = projectData(img_norm, U, K);
#imshow(Z); #Undocument to see pca

#putting all features into a row vector
#but does this make sense? After applying PCA in the way we did apply
for index = 1:total
    X_test((iter-8),index) = Z(index);
    end;
end;
predictions = svmPredict(model, X_test);
#model
disp(predictions);
##===========================================================================================================
