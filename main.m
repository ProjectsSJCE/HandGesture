#Set the sizes on the image and frames
max_row = 1000;
max_col = 1000;
frame_size = 70;

#Reads the Image
#img = imread ("hand.jpg");
img = imread ("blue.jpg");
#img = imread ("palm_white.jpg");

#Blackens the palm
[img] = blacken_palm(img, frame_size);
#imshow(img); #undocument to see blackened hand

#Filters the image
#should think of some good filter the remove speckles in the image. For now tried an averager filter. Didn't do much of a good job, so have documented it.
#filter_size = 5;
#[img] = filter_image(img, frame_size, filter_size);
#imshow(img); #Undocument to see filtered hand

#Have documented this currently because doing pca of 4900 coloumns took shit long
##As all the pixels are features, I have made one coloumn vector of all the pixels
#number = frame_size * frame_size;
#img_temp = zeros(number,1);
#for i = 1:number,
#    img_temp(i) = img(i);
#end;
#img_temp = img_temp'; #made it a row vector, each describing a different feature


#See, doing PCA this way won't work(this was the fault before i guess) because if is use m*m matrix(all pixels), pca will treat the first coloumn as the first feature type of the first m images, but we have one image and all the matrix values are it's features', so I did what i did in line 23 to 28, where i made a row vector of all the pixels and my comp crashed when doing PCA!!
#Is what I'm thinking correct??

#Does featureNormalize
[img_norm, mu, sigma] = featureNormalize(img);
#imshow(img_norm); #Undocument to see the normalised hand

#Apply PCA to get a feature vector
[U, S] = pca(img_norm);

#Get first 50 vectors. I'm going wrong here, something not right
#Let me retain 1000 pixels
#K = 1000;
K = 50;
Z = projectData(img_norm, U, K);
#imshow(Z); #Undocument to see the WRONG pca shit :P

#Doubt, should I retain the pixels as size(frame_size,1000/frame_size) or size(sqrt(1000),sqrt(1000))


##Neural networks
#input_layer_size  = 4900;  % 20x20 Input Images of Digits
#hidden_layer_size = 100;   % 25 hidden units
#num_labels = 2; 

#lambda = 1;

#J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
#                   num_labels, X, y, lambda);

#fprintf('\nInitializing Neural Network Parameters ...\n')

#initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
#initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

#% Unroll parameters
#initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
##nn_params = [Theta1(:) ; Theta2(:)];

#fprintf('\nTraining Neural Network... \n')

#%  After you have completed the assignment, change the MaxIter to a larger
#%  value to see how more training helps.
#options = optimset('MaxIter', 50);

#%  You should also try different values of lambda
#lambda = 1;

#% Create "short hand" for the cost function to be minimized
#costFunction = @(p) nnCostFunction(p, ...
#                                   input_layer_size, ...
#                                   hidden_layer_size, ...
#                                   num_labels, X, y, lambda);

#% Now, costFunction is a function that takes in only one argument (the
#% neural network parameters)
#[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

#% Obtain Theta1 and Theta2 back from nn_params
#Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
#                 hidden_layer_size, (input_layer_size + 1));

#Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
#                 num_labels, (hidden_layer_size + 1));
