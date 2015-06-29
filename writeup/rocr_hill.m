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

features = [];
remaining = 1:d;
err = 1;
mis = 0;

while length(remaining) > 0 && mis < 2
    addfive = tic;
    err_next = ones(1, d);
    for candidate=remaining
        features_p = [features candidate];
        [model,time_el]=LSPC_train(xtrain(:,features_p)', ytrain', 6.95, 1e-7);
        [yest junk] = LSPC_test(xtest(:,features_p)', model);
        err_next(candidate) = sum(yest'~=ytest) / length(ytest);
    end
    
    delta = err - err_next;
    
    [sv, si] = sort(delta, 'descend');
    next_incl = si(1:5);
    
    features = [features next_incl];
    remaining = setdiff(1:d, features);
    
    [model,time_el]=LSPC_train(xtrain(:,features)', ytrain');
    [yest junk] = LSPC_test(xtest(:,features)', model);
    err_old = err;
    err = sum(yest'~=ytest) / length(ytest);
    if (err_old - err < 0.001)
        mis = mis + 1;
    else
        mis = 0;
    end
    disp(sprintf('# feat: %d, err: %.4f, delta: %.4f', length(features), err, err_old - err));
    toc(addfive)
end

%% Prune (TODO)
while length(features) > 0
    err_next = ones(d, 1);
    for remove=features
        features_p = setdiff(features, remove);
        [model,time_el]=LSPC_train(xtrain(:,features_p)', ytrain');
        [yest junk] = LSPC_test(xtest(:,features_p)', model);
        err_next(remove) = sum(yest'~=ytest) / length(ytest);
    end
    
    idx = find(err_next == min(err_next));
    idx = idx(1);
    
    if (err_next(idx) > 0.001)
        break
    end

    features = setdiff(features, idx);
    [model,time_el]=LSPC_train(xtrain(:,features)', ytrain');
    [yest junk] = LSPC_test(xtest(:,features)', model);
    err_old = err;
    err = sum(yest'~=ytest) / length(ytest);
    disp(sprintf('Pruned %d, err: %.4f, delta: %.4f', idx, err, err_old - err));
end