function [Pre_Labels,time_taken]=multiclass_ntsvm_pairwise(train_data, train_target, test_data,c1,c2,c3)
%explanation...
%target = samples x class
%data = samples x (features + 1)
%R is rest class
[n,num_class] = size(train_target);

train_data = [train_data ones(n,1)];
test_data = [test_data ones(size(test_data,1),1)];
[~,f] = size(train_data);
time_taken = 0;
a = [];

pairwise_classifier = {};% cell_array initial
for i = 1 : num_class
    A = train_data(train_target(:,i)==1,:); %selected first class
    tic;
    fA = -inv(2*(c1*(A'*A) + c2*eye(f)) );
    time_fA = toc;
    for j = 1 : num_class
        if i==j
            continue
        end
        B = train_data(train_target(:,j)==1,:); % selected 2nd class
        m = size(B,1);
        tic;
        Z = fA * (c3*(B'*ones(m,1))/m) ;
        time = toc;
        a = 1/norm(Z(1:end-1,1));
        pairwise_classifier{i,j} = a*Z;
        clear Z a
        time_taken = time_taken+ time + time_fA;
    end
end
clear i j
%compute distance of train_data from hyperplanes
train_proj = {};  % this is test projection not train
for i=1: num_class
    for j= 1: num_class
        if i==j
            continue
        end
        train_proj{i,j} = abs((test_data*pairwise_classifier{i,j}));
%         train_proj = abs(train_proj);
    end
end
clear i j
labels = {};
%compare classes
for i=1:num_class
    for j=1: num_class
        if i==j
            continue
        end
        temp = sign(train_proj{j,i}-train_proj{i,j});
        temp(temp==-1) = 0;
        labels{i,j} = temp;
    end
end
clear temp
[st,~] = size(test_data); %size of test data = st
votes=zeros(st, num_class);
for i=1:num_class
    temp = 0;
    for j=1: num_class
        if i==j
            continue
        end
         temp = labels{i,j} + temp;
    end
    votes(:,i) = temp;
    clear temp
end

for i = 1:st
	[~,idx] = max(votes(i,:));
	y(i) = idx;
end

pre_label = zeros(st,num_class);
for i=1:st
    pre_label(i,y(i))= 1;
end
Pre_Labels = pre_label';
% pre_train_target = zeros(n,num_class);
% for i = 1 : n
%     [~,indx(i,:)] = min(train_proj(i,:));
%     pre_train_target(i,indx(i)) = 1;
% end
% clear indx
% 
% ntest = size(test_data,1);
% test_proj = (test_data*Z).*(repmat(a,ntest,1));
% test_proj = abs(test_proj);
% 
% pre_test_target = zeros(ntest,num_class);
% for i = 1 : ntest
%     [~,indx(i,:)] = min(test_proj(i,:));
%     pre_test_target(i,indx(i)) = 1;
% end
% 
% 
% Pre_Labels = pre_test_target';

end