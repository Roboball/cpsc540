function [B] = admm_enet( X, A, lambda1, lambda2, rho, dispPlot )
% X is the signal
% A is the dictionary
% lambda1 is the lasso penalty
% lambda2 is the ridge penalty
% rho is the augmented relaxation parameter
% dispPlot option displays plot likelihood over iterations

% return B, 

[n,p] = size(X);
[m,k] = size(A);

% max # of iterations, large to guarantee convergence
ITER = 1000;

% convergence sensitivity
atol = 1e-3;

%over-relaxation parameter to improve convergence?
alpha = 1.5;

%% variable initializations and pre-allocations
I = speye(k);
Bstar = randn(k,p);
Z = randn(k,p);
Y = zeros(k,p);

gamma = lambda1/(sqrt(1+lambda2));
X = (1/sqrt(1+lambda2)).*[X; zeros(k,p)];
A = [A;(sqrt(lambda2))*speye(k,k)];

%% Part 1: ELASTIC NET - solving for B
% Solve min_B 0.5|X-AB|_{2}^2 + lambda1|Z|_{1} + lambda2|Z|_{2}^2 s.t. Z = B

% we will exploit A'*A and use cholesky decomposition for computational efficiency
L = sparse(chol((A'*A + rho*I),'lower'));

cost=[];
% updates
for i=1:ITER
    % save old val to check for convergence
    B_old = Bstar;
    
    % solve Ly = b then L'*x = y
    Bstar = L' \ (L \ (A'*X + rho*Z - Y));
    
    % "found that over-relaxation with ? = 1.5 empirically
    % sped up convergence"
    Bstar = alpha*(Bstar) + (1-alpha)*Z;
        
    %Bstar = (A'*A + rho*I) \ (A'*X + rho*Z - Y);
    Z = soft(Bstar + (Y/rho), gamma/rho);
    Y = Y + rho*(Bstar - Z);
    
    cost(i)=obj(X,A,sqrt(1+lambda2).*Bstar,lambda1,lambda2);
    
    % check convergence ( an absolute tolerance )
    if (norm(Bstar - B_old) < atol)
        break
    end  
end

B = sqrt(1+lambda2).*Z;

if dispPlot == 1
    figure
    showPlot(cost);
end
end

%% define soft thresholding function
% if x > lambda then x - lambda, if x < lambda then x + lambda, else 0.

function y = soft(x, lambda)
y = sign(x).*max(abs(x) - lambda,0);
end

%% define objective function
function y = obj(X,A,B,lambda1, lambda2)
y = 0.5*(norm(X-(A*B),2).^2) + lambda1*(norm(B,1)) + lambda2*(norm(B,2).^2);
end

function showPlot(cost)
plot(1:size(cost,2),cost);
title('graph of likelihood per iteration')
xlabel('iteration') % x-axis label
ylabel('log-likelihood') % y-axis label
end
