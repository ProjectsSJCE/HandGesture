function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
%sigma = 0.3;
sigma = 0.1;
sig_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
C_array = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%min_vec = [0,0,0];
%min_value = 100;
%for i = 1:length(C_array),
%	for j = 1:length(sig_array),
%		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2));
%		predictions = svmPredict(model, Xval); 
%		cost = mean(double(predictions~= yval));
%		if (cost < min_value),
%			min_vec(1) = cost;
%			min_vec(2) = C_array(i);
%			min_vec(3) = sig_array(j);
%			min_value = cost;
%		end;		
%		disp("saar cost is for");
%		disp(C_array(i));
%		disp(sig_array(j));
%		disp(cost);
%	end;
%end;
%disp(min_vec);
%pause;


% 0.03 1 0.1


% =========================================================================

end
