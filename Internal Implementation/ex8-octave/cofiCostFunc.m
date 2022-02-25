function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                    num_features, lambda)
  %COFICOSTFUNC Collaborative filtering cost function
  %   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
  %   num_features, lambda) returns the cost and gradient for the
  %   collaborative filtering problem.
  %
  
  % Unfold the U and W matrices from params
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), ...
                  num_users, num_features);
  
              
  % You need to return the following values correctly
  J = 0;
  X_grad = zeros(size(X));            % Nm x n
  Theta_grad = zeros(size(Theta));    % Nu x n
  
  % ====================== YOUR CODE HERE ====================

  % Without Regularization 
  Error = (X*Theta') - Y;
  
  J = (1/2)*sum(sum(Error.^2.*R));
  
  X_grad = (Error.*R)*Theta;   % Nm x n
  Theta_grad = (Error.*R)'*X;  % Nu x n
  
  % With Regularization
  Reg_term_theta = (lambda/2)*sum(sum(Theta.^2));
  Reg_term_x = (lambda/2)*sum(sum(X.^2));
  
  J = J + Reg_term_theta + Reg_term_x;
  
  X_grad = X_grad + lambda*X;             % Nm x n
  Theta_grad = Theta_grad + lambda*Theta; % Nu x n
  
  % ==========================================================
  
  grad = [X_grad(:); Theta_grad(:)];

end