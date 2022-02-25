function [J, grad] = costFunction(theta, X, y)
  %COSTFUNCTION Compute cost and gradient for logistic regression
  %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
  %   parameter for logistic regression and the gradient of the cost
  % Initialize some useful values
  m = length(y); 
  J = 0;
  grad = zeros(size(theta));
  
  % ====================== YOUR CODE HERE ======================
    
  z = X * theta;      
  h_x = sigmoid(z);    
  
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))); 
  
  grad = (1/m)* (X'*(h_x-y));     
  
  % =============================================================
  
end