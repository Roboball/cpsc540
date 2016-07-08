clear all

data = load('housing.data');
X = data(:,1:end-1);
y = data(:,end);

% Standardize features to have mean of 0 and variance of 1
X = standardize(X);

% Add bias variable
N = size(X,1);
X = [ones(N,1) X];
d = size(X,2);

lambdaValues = 2.^[16:-1:4];
for i = 1:length(lambdaValues)
    lambda = lambdaValues(i);
    w_lasso(:,i) =admm_enet(y,X,lambda,0,4,0);
    %w_ridge(:,i) =admm_enet(y,X,0,lambda,4,0);
    mod =ridgeReg(y,X,lambda);
    w_ridge(:,i) = mod.B;
    w_enet(:,i) =admm_enet(y,X,lambda,lambda,4,0);
end
figure
plot(log2(lambdaValues),w_lasso);
title('Regularization Path - Lasso');
xlabel('log2(lambda)');
ylabel('w coefficients');

figure
plot(log2(lambdaValues),w_ridge);
title('Regularization Path - Ridge');
xlabel('log2(lambda)');
ylabel('w coefficients');

figure
plot(log2(lambdaValues),w_enet);
title('Regularization Path - Elastic Net');
xlabel('log2(lambda)');
ylabel('w coefficients');

% rhoValues = 2.^[16:-1:4];
% for i = 1:length(rhoValues)
%     rho = rhoValues(i);
%     w_lasso(:,i) =admm_enet(y,X,3.3,0,rho,0);
%     w_ridge(:,i) =admm_enet(y,X,0,3.3,rho,0);
%     w_enet(:,i) =admm_enet(y,X,3.3,3.3,rho,0);
% end
% 
% figure
% plot(log2(rhoValues),w_lasso);
% title('Rho Path - Lasso');
% xlabel('log2(rho)');
% ylabel('w coefficients');
% 
% figure
% plot(log2(rhoValues),w_ridge);
% title('Rho Path - Ridge');
% xlabel('log2(rho)');
% ylabel('w coefficients');
% 
% figure
% plot(log2(rhoValues),w_enet);
% title('Rho Path - Elastic Net');
% xlabel('log2(rho)');
% ylabel('w coefficients');