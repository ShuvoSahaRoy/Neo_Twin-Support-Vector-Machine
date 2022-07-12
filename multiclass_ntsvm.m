function [Pre_Labels,time_taken]=multiclass_ntsvm(train_data, train_target, test_data,c1,c2,c3)
%explanation...
% binary relevence approch 
%target = samples x class
%data = samples x (features + 1)
%R is rest class
[n,num_class] = size(train_target);

train_data = [train_data ones(n,1)];
test_data = [test_data ones(size(test_data,1),1)];
[~,f] = size(train_data);
time_taken = 0;
a = [];
for i = 1 : num_class   
    A = train_data(train_target(:,i)==1,:);
    R = train_data(train_target(:,i)~=1,:);
    m = size(R,1);
    tic;
    Z(:,i) = -inv(2*(c1*(A'*A) + c2*eye(f)) ) * (c3*(R'*ones(m,1))/m) ;
    time = toc;
    time_taken = time_taken+ time;
    a(i) = 1/norm(Z(1:end-1,i));
end

%compute distance of train_data from hyperplanes
train_proj = (train_data*Z).*(repmat(a,n,1));
train_proj = abs(train_proj);

pre_train_target = zeros(n,num_class);
for i = 1 : n
    [~,indx(i,:)] = min(train_proj(i,:));
    pre_train_target(i,indx(i)) = 1;
end
clear indx

ntest = size(test_data,1);
test_proj = (test_data*Z).*(repmat(a,ntest,1));
test_proj = abs(test_proj);

pre_test_target = zeros(ntest,num_class);
for i = 1 : ntest
    [~,indx(i,:)] = min(test_proj(i,:));
    pre_test_target(i,indx(i)) = 1;
end


Pre_Labels = pre_test_target';

end
