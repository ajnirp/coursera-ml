function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

errors = X*theta - y;
theta_copy = theta(2:end);

main_cost = (errors' * errors) / (2*m);
reg_cost = (theta_copy' * theta_copy * lambda) / (2*m); 

J = main_cost + reg_cost;

grad = (lambda/m)*theta + X' * errors / m;
grad(1) = grad(1) - (lambda/m)*theta(1);

% =========================================================================

grad = grad(:);

end
