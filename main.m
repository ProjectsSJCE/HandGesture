#Set the sizes on the image and frames
max_row = 1000;
max_col = 1000;
frame_size = 100;

#Reads the Image
#img = imread ("hand.jpg");
img = imread ("blue.jpg");
#img = imread ("palm_white.jpg");

#Blackens the palm
[img] = blacken_palm(img, frame_size);
imshow(img); undocument to see blackened hand

#Filters the image
#should think of some good filter the remove speckles in the image. For now tried an averager filter. Didn't do much of a good job, so have documented it.
#filter_size = 5;
#[img] = filter_image(img, frame_size, filter_size);
#imshow(img); Undocument to see filtered hand

#Does featureNormalize
[img_norm, mu, sigma] = featureNormalize(img);
#imshow(img_norm); Undocument to see the normalised hand

#Apply PCA to get a feature vector
[U, S] = pca(img_norm, frame_size, frame_size);

#Get first 50 vectors. I'm going wrong here, something not right
K = 50;
Z = projectData(img_norm, U, K);
#imshow(Z);
#should be something other than Z i guess

#imshow(U(:, 1:50)');

#K = 100;
#X_rec  = recoverData(Z, U, K);
#imshow(X_rec);
