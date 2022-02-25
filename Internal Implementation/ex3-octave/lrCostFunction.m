function [J, grad] = lrCostFunction(theta, X, y, lambda)
  %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
  %regularization
  %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  J = 0;
  grad = zeros(size(theta));
  
  % ==========================================================
  z   = X * theta;   
  h_x = sigmoid(z);  
  
  reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
  
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term; 
  
  grad(1) = (1/m) * (X(:,1)'*(h_x-y));                                    
  grad(2:end) = (1/m) * (X(:,2:end)'*(h_x-y)) + (lambda/m)*theta(2:end);  
  
  % ==========================================================
  
  grad = grad(:);
end