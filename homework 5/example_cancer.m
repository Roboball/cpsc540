%% load, split into training and testing
clear all

data = load('prostate.data');
X = data(:,1:end-1);
y = data(:,end);

N = size(X,1);
d = size(X,2);

ytrain = y(1:50);
ytest = y(51:end);
Xtrain = X(1:50,:);
Xtest = X(51:end,:);

Xbar = mean(Xtrain);
Xstd = std(Xtrain);
ybar = mean(ytrain);

ytrain = ytrain - ybar;
Xtrain = standardize(Xtrain);

%% compute lambda paths for ridge
lambdaValues = logspace(-1.5, 3.5,20);
for i = 1:length(lambdaValues)
    lambda = lambdaValues(i);
    w_lasso(:,i) =admm_enet(y,X,lambda,0,4,0);
    w_ridge(:,i) =admm_enet(y,X,0,lambda,4,0);
    w_enet(:,i) =admm_enet(y,X,lambda,lambda,4,0);
end

%% plots
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

%% cross-validation
w = [];
w0 = [];
for lambda=lambdaValues
    w = [w admm_enet(ytrain,Xtrain,0,lambda,4,0)];
end

trainerror = [];
testerror = [];
min_err = 0;

for i=1:size(w,2)
    wi = w(:,i);
    yhatstest = ybar + (Xtest- repmat(Xbar,size(Xtest,1),1))./repmat(Xstd,size(Xtest,1),1)*wi;
    yhatstrain = ybar + (Xtrain)*wi;
    trainerror= [trainerror norm((ytrain + ybar)-yhatstrain, 2)./ norm(ytrain + ybar, 2)];
    testerror= [testerror norm(ytest - yhatstest,2) ./norm(ytest ,2)];
    max_err = max(trainerror(end), testerror(end));
    if (min_err == 0 || max_err < min_err)
        min_err = max_err;
        best_lambda = lambdaValues(i);
    end
end
fprintf('best lambda: %d\n', best_lambda);

figure
plot(log(lambdaValues),trainerror,'-bo','LineWidth',2);

hold on
plot(log(lambdaValues),testerror,'-g^','LineWidth',2);

xlabel('lambda'); ylabel('error')
legend('Train', 'Test')