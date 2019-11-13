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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%implementing forward propagation to calculate h(x)
% spinach mozrella

a1 = [ones(m,1), X];

z2 = a1 * Theta1';
a2 = sigmoid(z2) ;
a2 = [ones(m,1),a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

hx = a3;

% make a vector of y(i)k ==> this will be a 5000 x 10 matrix
yk = zeros(m, num_labels);
for i=1:m
    temp = zeros(1,num_labels);
    temp(y(i)) = 1;
    yk(i, :) = temp;
end

temp = (yk .* log(hx) + (1-yk) .* log(1-hx)); % this is 5000 x 10 matrix 

% we need to find the sum of temp (5000 x 10)
temp = sum(sum(temp));
J_unreg = -(1/m) * temp ;

temp_Theta1 = Theta1;
temp_Theta1(:,1) = 0;
    
temp_Theta2 = Theta2;
temp_Theta2(:,1) = 0;

        
reg_param = lambda/(2*m) * ( sum(sum(temp_Theta1 .^ 2)) + sum(sum(temp_Theta2 .^ 2)) ) ;

J = J_unreg + reg_param;

% -------------------------------------------
% Computing  gradients using back propagation
% -------------------------------------------

%  Theta1 ==> 25 X 401 vector
%  Theta2 ==> 10 X 26 vector

for i=1:m
    % propagting forward to calculate the error in the outut layer
    a1 = [1, X(i,:)]';                  % Transpose of a1 makes it ==> 401 X 1 vector
    
    z2 = Theta1 * a1;                   % 25 X 1 vector
    a2 = sigmoid(z2);
    a2 = [1; a2];
    
    z3 = Theta2 * a2;                   % 10 x 1 vector
    a3 = sigmoid(z3);  
    
    % Calculating the Errors 
    delta_3 =  a3 - yk(i, :)';              % 10 x 1 vector
    
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2]); 
    delta_2 = delta_2(2:end);           % 25 X 1 vector
    
    Theta2_grad = Theta2_grad + (delta_3 * a2');    % 10 X 26 Vector
    Theta1_grad = Theta1_grad + (delta_2 * a1');    % 10 X 26 Vector
  
end

Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% Add regularization terms

temp_Theta1 = Theta1;
temp_Theta2 = Theta2;

temp_Theta1(:, 1) = 0;          % we dont regularize bias unit
temp_Theta2(:, 1) = 0;          % we dont regularize bias unit

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * temp_Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * temp_Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
