function  [Pre_Label, Testacc, Trainacc, time, z1, z2] = Neo_TSVM(Xtrain, Ytrain, Xtest, Ytest, c1, c2, c3, c4)

%%explanation
%Xtrain dimension shd be n x f
%Ytrain dimension shd be n x 1, it shd have entries +1 or -1
%class A has points which hv label +1
%class B has points which hv label -1

D = Xtrain;
A = D(Ytrain==1,:);
B = D(Ytrain~=1,:);
m1 = size(A,1);
m2 = size(B,1);

A = [A ones(m1,1)];
B = [B ones(m2,1)];
D = [D ones(m1+m2,1)];
Xtest = [Xtest, ones(size(Xtest,1),1)];

f = size(D,2);
%tol_val = 10e-4;
% iter = 0;
%max_iter = 50;
%max_iter = 2;

%diff =1000;
%...................
tic;
%I = 2*c3*eye(f);
% denomin1_inv = inv(2 * c1*(A'*A) + I);
% denomin2_inv = inv(2 * c1*(B'*B) + I);
meanA=A'*(ones(m1,1))/m1;
meanB=B'*(ones(m2,1))/m2;
time1=toc;

%.................
tic;
%z1 = -(2 * c1*(A'*A) + I) \ meanB;
%z2 = -(2 * c1*(B'*B) + I) \ (meanA + c2*z1);

z2 = -inv(2 * (c1*(B'*B) + c3*eye(f))) * (c4*meanA);
%c4=0;
z1 = -inv(2 * (c1*(A'*A) + c3*eye(f))) * (c4*meanB + c2*z2);
time2 = toc;
%................

time = time1 + time2;

proj1 = (D*z1)/norm(z1(1:end-1));
proj2 = (D*z2)/norm(z2(1:end-1));

t=abs(proj2) - abs(proj1);
train_Label = sign( abs(proj2) - abs(proj1) );

Trainacc = length(find(train_Label==Ytrain))/ length(Ytrain)*100;

clear proj1 proj2

proj1 = Xtest*z1/norm(z1(1:end-1));
proj2 = Xtest*z2/norm(z2(1:end-1));

Pre_Label = sign( abs(proj2) - abs(proj1) );

Testacc = length(find(Pre_Label==Ytest))/ length(Ytest)*100;

end
