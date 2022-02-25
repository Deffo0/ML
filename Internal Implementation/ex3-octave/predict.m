function p = predict(Theta1, Theta2, X)
  %PREDICT Predict the label of an input given a trained neural network
  %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
  %   trained weights of a neural network (Theta1, Theta2)
  
  m = size(X, 1);
  num_labels = size(Theta2, 1);
  
  p = zeros(size(X, 1), 1);  % m x 1
  
    % ========================================================
      
  a1 = [ones(m,1) X]; 
  z2 = a1 * Theta1';  % 5000 x 25
  a2 = sigmoid(z2);   % 5000 x 25
 
  a2 =  [ones(size(a2,1),1) a2];  % 5000 x 26
  
  z3 = a2 * Theta2';  % 5000 x 10
  a3 = sigmoid(z3);  % 5000 x 10
  
  [prob, p] = max(a3,[],2); 
    % ========================================================
end