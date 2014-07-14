function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
X = [ones(m, 1) X];
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
#Size of theta1 = 25 X 401
#Size of theta2 = 10 X 26
z2 = X * Theta1'; #5000X25
a2 = sigmoid(z2);
nl = size(a2, 1);
a2 = [ones(nl, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3); #size 5000 X 10 matrix y is 5000 X 1

hypothesis = a3;
#This computes unregularized
#J_values = zeros(1, num_labels);
#for iter = 1:num_labels,
#    hyp = hypothesis(:, iter);
#    z = (y==iter);
#    J_old = - (z'*log(hyp) + (1-z')*log(1 - hyp)) / m;
#    J_values(iter) = J_old;
#    end;
#J = sum(J_values);
sub_matrix = zeros(m,num_labels);
J_values = zeros(1, num_labels);

for iter = 1:num_labels,
    hyp = hypothesis(:, iter);
    z = (y==iter);
    J_old = - (z'*log(hyp) + (1-z')*log(1 - hyp)) / m;
    J_values(iter) = J_old;
    sub_matrix(:,iter) = z; #5000x1 finally becomes 5000x10
    end;
J = sum(J_values);

sub_matrix = sub_matrix'; #10X5000

#with regularization
J += lambda * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))) / (2*m);

#Sharath logic
#J += lambda * (sum(sum(Theta1(26:end) .^ 2)) + sum(sum(Theta2(11:end) .^ 2))) / (2*m); 

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
a3 = a3';
a2 = a2';
a1 = X';
#a3 = 10 X 5000
#a2 = 26X5000
delta3 = zeros(num_labels, m);


#for t = 1:m,
#   delta3(:, t) = a3(:, t);
#    delta3(y(t), t) -= 1;
#    delta2(:, t) = ((Theta2)'*delta3(:, t)) .* (a2(:,t) .* (1 - a2(:, t)));
#    temp = delta2(:, t);
#    temp = temp(2:end);
#    Theta1_grad += temp * (a1(:,t))';
#    Theta2_grad += delta3(:, t) * (a2(:,t))'; 
#end;

delta3 = a3 - sub_matrix;
delta2 = (((Theta2)'*delta3) .* (a2 .* (1 - a2)));
temp = delta2(2:end,:);
Theta1_grad = temp * (a1');
Theta2_grad = delta3 * (a2');
Theta1_grad /= m;
Theta2_grad /= m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) .+ ((lambda * (sum(sum(Theta1(:, 2:end))))) / m); 
%Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) .+ ((lambda * (sum(sum(Theta2(:, 2:end))))) / m);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda * (Theta1(:, 2:end))) / m); 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda * (Theta2(:, 2:end))) / m);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
