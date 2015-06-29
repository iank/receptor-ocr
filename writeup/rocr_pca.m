rocr = csvread('rocr.csv');

rocr = rocr(randperm(size(rocr, 1)),:);
n = size(rocr, 1);
d = size(rocr, 2) - 1;

X = rocr(:, 1:d);
Y = rocr(:, d+1);

%nz = X > 0.00001;
%z = X <= 0.00001;
%X(nz) = 1;
%X(z) = 0;

n_tr = floor(0.75*n);

xtrain = X(1:n_tr,:);
ytrain = Y(1:n_tr);
xtest = X(n_tr+1:end,:);
ytest = Y(n_tr+1:end);

%%

mu = mean(xtrain);
phi = xtrain - repmat(mu, [size(xtrain,1) 1]);
C = phi'*phi;

[V, D] = eig(C);

%%

n_r = []
err_r = []
for n_eig=1:d
    % Columns of V are eigenvectors, and D is reverse-sorted
    U = V(:,d-(n_eig-1):end);
    xtrain_pca = (xtrain - repmat(mu, [size(xtrain,1) 1])) * U;
    xtest_pca = (xtest - repmat(mu, [size(xtest,1) 1])) * U;
    [model,time_el]=LSPC_train(xtrain_pca', ytrain');
    [yest junk] = LSPC_test(xtest_pca', model);

    err = sum(yest'~=ytest) / length(ytest);
    disp(sprintf('LSPC: n_eig=%d, %.4f test err', n_eig, err)); 
    n_r = [n_r n_eig];
    err_r = [err_r err];
end