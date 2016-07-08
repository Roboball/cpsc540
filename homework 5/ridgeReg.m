function [model] = ridgeReg( X, A, lambda )
% X is the signal
% A is the dictionary
% lambda is the ridge penalty
% dispPlot option displays plot likelihood over iterations

% return model

%% variable initializations and pre-allocations
B = ridge(X,A, lambda); 
model.B = B;
model.cost = obj(X,A,B,lambda);
model.lambda = lambda;

end

%% regress on ridge coefficients
function B = ridge(X,A,lambda)
I = speye(size(A,2));
L = sparse(chol((A'*A + lambda*I),'lower'));
% solve Ly = b then L'*x = y
B = L' \ (L \ (A'*X));
end

%% define objective function
function y = obj(X,A,B,lambda)
y = 0.5*(norm(X-(A*B),2).^2) + lambda*(norm(B,2).^2);
end