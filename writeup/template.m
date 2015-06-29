%rocr = csvread('train_template.csv');
% MATLAB csvread breaks on long lines (over 100,000 columns) and returns a
% single column of data

C = textread('train_template.csv', '%s', 'delimiter', '\n', 'bufsize', 600000);
lines = size(C, 1);
cols = size(cell2mat(textscan(C{1}, '%d', 'delimiter', ',')), 1);

rocr = zeros(lines, cols);
for k=1:lines
	rocr(k,:) = cell2mat(textscan(C{k}, '%d', 'delimiter', ','))';
end

%% 

rocr = rocr(randperm(size(rocr, 1)),:);
n = size(rocr, 1);
d = size(rocr, 2) - 1;

X = rocr(:,1:d);
Y = rocr(:,d+1);

n_tr = floor(0.75*n);

xtrain = X(1:n_tr,:);
ytrain = Y(1:n_tr);
xtest = X(n_tr+1:end,:);
ytest = Y(n_tr+1:end);

%%

[model,time_el]=LSPC_train(xtrain', ytrain');
[yest junk] = LSPC_test(xtest', model);

err = sum(yest'~=ytest) / length(ytest);
disp(sprintf('LSPC: %.4f test err', err)); 
