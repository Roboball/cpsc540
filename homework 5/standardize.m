function [ S, mu , sigma ] = standardize( X )
%Function to standardize columns of feature matrix
%   rescaled to have properties of a standard normal distribution, i.e.
%   mean of zero and standard deviation of 1.

mu = repmat(mean(X),size(X,1),1);
sigma = repmat(std(X),size(X,1),1);
S = (X - mu)./sigma;

end

