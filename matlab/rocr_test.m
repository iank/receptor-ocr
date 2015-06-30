rocr = csvread('rocr.csv');

rocr = rocr(randperm(size(rocr, 1)),:);
n = size(rocr, 1);
d = size(rocr, 2) - 1;

X = rocr(:, 1:d);
Y = rocr(:, d+1);

nz = X > 0.00001;
z = X <= 0.00001;
X(nz) = 1;
X(z) = 0;

n_tr = floor(0.75*n);

xtrain = X(1:n_tr,:);
ytrain = Y(1:n_tr);
xtest = X(n_tr+1:end,:);
ytest = Y(n_tr+1:end);

%%

n_r = []
err_r = []
for n_receptors=1:d
    [model,time_el]=LSPC_train(xtrain(:,1:n_receptors)', ytrain');
    [yest junk] = LSPC_test(xtest(:,1:n_receptors)', model);

    err = sum(yest'~=ytest) / length(ytest);
    disp(sprintf('LSPC: n_r=%d, %.4f test err', n_receptors, err)); 
    n_r = [n_r n_receptors];
    err_r = [err_r err];
end
