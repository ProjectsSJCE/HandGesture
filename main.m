#Set the sizes on the image and frames
max_row = 1000;
max_col = 1000;
frame_size = 100;

#Reads the Image
img = imread ("hand.jpg");
#img = imread ("blue.jpg");
#img = imread ("palm_white.jpg");
#size(img)

#Blackens the palm
[img] = blacken_palm(img, frame_size);
#imshow(img); #undocument to see blackened hand
##pause;

##Filters the image

#frame_size = 80;
img = crop_hand(img, frame_size);
#imshow(img);

#Applying a high pass filter, since it is used for edge detection, seems to do a better job
#high_pass = [0.25,0.5,0.25;0.5,9,0.5;0.25,0.5,0.25];
#img = imfilter(img, high_pass, "replicate");
#imshow(img); #Undocument to see filtered hand

#Does featureNormalize
[img_norm, mu, sigma] = featureNormalize(img);
imshow(img_norm); #Undocument to see the normalised hand
#pause;

disp("Now starting pca");
[U, S] = pca(img_norm, frame_size, frame_size);
disp("Done with pca");
K = 30;
Z = projectData(img_norm, U, K);
#imshow(Z); #Undocument to see pca

total = frame_size * K;

X = zeros(2, total);

#Have to decide on C and sigma values
C = 1;
%sigma = 0.3;
sigma = 0.1;

#putting all features into a row vector
#but does this make sense? After applying PCA in the way we did apply
for index = 1:total
    X(1,index) = Z(index);
    X(2,index) = Z(index) + 0.5;
    end;
y = [1;0]; 
#X = [1,2;3,4];   
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2,sigma));
predictions = svmPredict(model, X);
disp(predictions);
##===========================================================================================================
