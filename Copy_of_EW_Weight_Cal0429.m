
%% 这个文档用于计算TW最优权重，游离于整个模型之外。


%% 导入数据
%clear;
fullSample = readmatrix('D:\OneDrive - 北京大学经济学院\房产税问题研究\新工作进展\full_sample.xls');
l_full = length(fullSample);


%% 不区分高低金融素养

%% 全样本估计
mySample = fullSample;
l = length(mySample);

% 计算权重
x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2./1000 , mySample(:,4) , mySample(:,4).^2./1000, ...,
     mySample(:,6),  mySample(:,7)  , mySample(:,8)];
% x里包括 cons t t^2 cash cash^2 housing housing^2 Ipart treat*post treat post 
k = size(x,2);
qq = zeros(k,k);
for i=1:l
    qq = qq + 1/l * x(i,:)'*x(i,:);
end
Q = inv(qq);


% y1:消费 考虑第一组beta
lamda1 = zeros(l,k);
y1 = mySample(:,1);
beta1 = (x'*x)\x'*y1;

for i=1:l
lamda1(i,:) = (Q*x(i,:)'*(y1(i,:)-x(i,:)*beta1)*sqrt(((l-1)/(l-k))))';
end

% y2:参与 考虑第二组beta
lamda2 = zeros(l,k);
y2 = mySample(:,2);
beta2 =(x'*x)\x'*y2;

for i=1:l
lamda2(i,:) = (Q*x(i,:)'*(y2(i,:)-x(i,:)*beta2)*sqrt(((l-1)/(l-k))))';
end

% y3:risk ratio 考虑第3组beta
lamda3 = zeros(l,k);
y3 = mySample(:,11);
beta3 =(x'*x)\x'*y3;

for i=1:l
lamda3(i,:) = (Q*x(i,:)'*(y3(i,:)-x(i,:)*beta3)*sqrt(((l-1)/(l-k))))';
end



% y4:fin ratio 考虑第4组beta
lamda4 = zeros(l,k);
y4 = mySample(:,10);
beta4 =(x'*x)\x'*y4;

for i=1:l
lamda4(i,:) = (Q*x(i,:)'*(y4(i,:)-x(i,:)*beta4)*sqrt(((l-1)/(l-k))))';
end



lamda = [lamda1 , lamda2 , lamda3 , lamda4];

phiphi = zeros(4*k,4*k,l);
for i = 1:l
phiphi(:,:,i) = lamda(i,:)' * lamda(i,:);
end
omega = sum(phiphi,3)/10;

WW = inv(omega);

%WW([9,10,19,20,29,30,39,40],:)=[];
%WW(:,[9,10,19,20,29,30,39,40])=[];
%WW(WW(:,:)>1e4)=WW(WW(:,:)>1e4)./100;
%WW(WW(:,:)<-1e4)=WW(WW(:,:)<-1e4)./100;
WW([7,8,15,16,23,24,31,32],:)=[];
WW(:,[7,8,15,16,23,24,31,32])=[];

save Sample_fullW.mat  WW  









%% 计算beta
fullSample = readmatrix('D:\OneDrive - 北京大学经济学院\房产税问题研究\新工作进展\full_sample.xls');
l=length(fullSample);
%% 2010
mySample = fullSample(fullSample(:,8)==0,:);
save mySample_pre10.mat mySample
%% 实验前 pre
mySample = fullSample(fullSample(:,8)==0|fullSample(:,7)==0,:);
l = length(mySample);

% 计算权重
x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) , mySample(:,4).^2 , ...,
     mySample(:,6) ,  mySample(:,7)  , mySample(:,8)];
% x里包括 cons t t^2 cash cash^2 housing housing^2 Ipart treat*post treat post 
k = size(x,2);
lamda1 = zeros(l,k);
lamda2 = zeros(l,k);
lamda3 = zeros(l,k);
lamda4 = zeros(l,k);
phiphi = zeros(4*k,4*k,l);

qq = zeros(k,k);
for i=1:l
    qq = qq + 1/l * x(i,:)'*x(i,:);
end
Q = inv(qq);


% y1:消费 考虑第一组beta
y1 = mySample(:,1);
beta1 = (x'*x)\x'*y1;
for i=1:l
lamda1(i,:) = (Q*x(i,:)'*(y1(i,:)-x(i,:)*beta1)*sqrt(((l-1)/(l-k))))';
end

% y2:参与 考虑第二组beta
y2 = mySample(:,2);
beta2 =(x'*x)\x'*y2;
for i=1:l
lamda2(i,:) = (Q*x(i,:)'*(y2(i,:)-x(i,:)*beta2)*sqrt(((l-1)/(l-k))))';
end

% y3: 考虑第3组beta
y3 = mySample(:,11);
beta3 =(x'*x)\x'*y3;
for i=1:l
lamda3(i,:) = (Q*x(i,:)'*(y3(i,:)-x(i,:)*beta3)*sqrt(((l-1)/(l-k))))';
end

% y4: 考虑第4组beta
y4 = mySample(:,10);
beta4 =(x'*x)\x'*y4;
for i=1:l
lamda4(i,:) = (Q*x(i,:)'*(y4(i,:)-x(i,:)*beta4)*sqrt(((l-1)/(l-k))))';
end

lamda_pre = [lamda1 , lamda2 , lamda3 , lamda4];


for i = 1:l
phiphi(:,:,i) = lamda_pre(i,:)' * lamda_pre(i,:);
end
omega = sum(phiphi,3)/l;

W = inv(omega);
W1 = W;
W = WW;
beta1([7,8])=[];
beta2([7,8])=[];
beta3([7,8])=[];
beta4([7,8])=[];

save Sample_pre.mat mySample W x beta1 beta2 beta3






%% 实验后 
mySample = fullSample(fullSample(:,8)==1&fullSample(:,7)==1,:);
l = length(mySample);

% 计算权重
x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) , mySample(:,4).^2 , ...,
       mySample(:,6)  ];


% x里包括 cons t t^2 cash cash^2 housing housing^2 Ipart treat*post treat post 
k = size(x,2);
lamda1 = zeros(l,k);
lamda2 = zeros(l,k);
lamda3 = zeros(l,k);
phiphi = zeros(3*k,3*k,l);

%Q = inv(x'*x);
qq = zeros(k,k);
for i=1:l
    qq = qq + 1/l * x(i,:)'*x(i,:);
end
Q = inv(qq);


% y1:消费 考虑第一组beta
y1 = mySample(:,1);
beta4 = (x'*x)\x'*y1;

for i=1:l
lamda1(i,:) = (Q*x(i,:)'*(y1(i,:)-x(i,:)*beta4)*sqrt(((l-1)/(l-k))))';
end

% y2:参与 考虑第二组beta
y2 = mySample(:,2);
beta5 =(x'*x)\x'*y2;

for i=1:l
lamda2(i,:) = (Q*x(i,:)'*(y2(i,:)-x(i,:)*beta5)*sqrt(((l-1)/(l-k))))';
end

% y3: 考虑第3组beta
y3 = mySample(:,11);
beta6 =(x'*x)\x'*y3;
for i=1:l
lamda3(i,:) = (Q*x(i,:)'*(y3(i,:)-x(i,:)*beta6)*sqrt(((l-1)/(l-k))))';
end

lamda_post = [lamda1 , lamda2 , lamda3];


for i = 1:l
phiphi(:,:,i) = lamda_post(i,:)' * lamda_post(i,:);
end
omega = sum(phiphi,3)/l;

W = inv(omega);
W2 = W;
W = WW;
save Sample_post.mat mySample W x beta4 beta5 beta6



%% 前+后 prepost
mySample = fullSample;
l = length(mySample);

ss = size(WW,1);
W = [WW zeros(ss,ss); WW zeros(ss,ss)];

save Sample_prepost.mat mySample W beta1 beta2 beta3 beta4 beta5 beta6



%% 前+后 prepost1
%{
mySample = fullSample(fullSample(:,7)==1,:);
l = length(mySample);

x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2./1000 , mySample(:,4) , mySample(:,4).^2./1000, ...,
     mySample(:,5) , mySample(:,5).^2./1000 , mySample(:,6)];

%x = [ones(l,1) , mySample(:,3) ,   mySample(:,4)  , ...,
%     mySample(:,5) ,  mySample(:,6)];

y1 = mySample(:,1);
beta1 = (x'*x)\x'*y1;

y2 = mySample(:,2);
beta2 =(x'*x)\x'*y2;

W = WW;

save Sample_prepost1.mat mySample W beta1 beta2 
%}



%% pre1
%{
fullSample = readmatrix('D:\OneDrive - 北京大学经济学院\房产税问题研究\新工作进展\full_sample.xls');

 
mySample = fullSample(fullSample(:,8)==0&fullSample(:,7)==1,:);
l = length(mySample);

% 计算权重
x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2./1000 ,  mySample(:,4) , mySample(:,4).^2./1000 ];

% x里包括 cons t t^2 cash cash^2 housing housing^2 Ipart treat*post treat post 
k = size(x,2);
lamda1 = zeros(l,k);


%Q = inv(x'*x);
qq = zeros(k,k);
for i=1:l
    qq = qq + 1/l * x(i,:)'*x(i,:);
end
Q = inv(qq);


% y1:消费 考虑第一组beta
y1 = mySample(:,1);
beta1 = (x'*x)\x'*y1;

for i=1:l
lamda1(i,:) = (Q*x(i,:)'*(y1(i,:)-x(i,:)*beta1)*sqrt(((l-1)/(l-k))))';
end

k1 = k;


% y2:参与 考虑第二组beta
x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2./1000 ,  mySample(:,4) , mySample(:,4).^2./1000 ];
k = size(x,2);
lamda2 = zeros(l,k);
phiphi = zeros(k1+k,k1+k,l);

qq = zeros(k,k);
for i=1:l
    qq = qq + 1/l * x(i,:)'*x(i,:);
end
Q = inv(qq);

y2 = mySample(:,2);
beta2 =(x'*x)\x'*y2;

for i=1:l
lamda2(i,:) = (Q*x(i,:)'*(y2(i,:)-x(i,:)*beta2)*sqrt(((l-1)/(l-k))))';
end

lamda = [lamda1 , lamda2];


for i = 1:l
phiphi(:,:,i) = lamda(i,:)' * lamda(i,:);
end
omega = sum(phiphi,3)/l*2  ;

W = inv(omega);
%W= WW;

save Sample_pre1.mat mySample W beta1 beta2  
%}




%% DID系数和权重
mySample = readmatrix('D:\OneDrive - 北京大学经济学院\房产税问题研究\新工作进展\full_sample.xls');
l = length(mySample);

%x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2. , mySample(:,4) ,  mySample(:,4).^2  , ...,
%     mySample(:,5) , mySample(:,5).^2  , mySample(:,6) , mySample(:,7), mySample(:,8), mySample(:,7).*mySample(:,8) ];

x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2./1000  , mySample(:,4) ,  mySample(:,4).^2./1000  , ...,
      mySample(:,6) , mySample(:,7), mySample(:,8), mySample(:,7).*mySample(:,8) ];


k = size(x,2);
lamda1 = zeros(l,k);
lamda2 = zeros(l,k);
lamda3 = zeros(l,k);
lamda4 = zeros(l,k);
qq = zeros(k,k);
phiphi = zeros(4*k,4*k,l);

for i=1:l
    qq = qq + 1/l * x(i,:)'*x(i,:);
end
Q = inv(qq);

% y1:消费 考虑第一组beta
y1 = mySample(:,1);
beta1 = (x'*x)\x'*y1;

for i=1:l
lamda1(i,:) = (Q*x(i,:)'*(y1(i,:)-x(i,:)*beta1)*sqrt(((l-1)/(l-k))))';
end

% y2:参与 考虑第二组beta
y2 = mySample(:,2);
beta2 = (x'*x)\x'*y2;

for i=1:l
lamda2(i,:) = (Q*x(i,:)'*(y2(i,:)-x(i,:)*beta2)*sqrt(((l-1)/(l-k))))';
end



% y3: 考虑第3组beta
y3 = mySample(:,11);
beta3 = (x'*x)\x'*y3;
for i=1:l
lamda3(i,:) = (Q*x(i,:)'*(y3(i,:)-x(i,:)*beta3)*sqrt(((l-1)/(l-k))))';
end

% y4: 考虑第4组beta
y4 = mySample(:,10);
beta4 = (x'*x)\x'*y4;
for i=1:l
lamda4(i,:) = (Q*x(i,:)'*(y4(i,:)-x(i,:)*beta4)*sqrt(((l-1)/(l-k))))';
end




lamda = [lamda1 , lamda2 , lamda3 , lamda4];


for i = 1:l
phiphi(:,:,i) = lamda(i,:)' * lamda(i,:) ;
end
omega = sum(phiphi,3)/10;


W = inv(omega);

W([7,8,16,17,25,26,34,35],:)=[];
W(:,[7,8,16,17,25,26,34,35])=[];

%{
W([9,10,11,12,13,26,27,28,29,30,43,44,45,46,47],:)=[];
W(:,[9,10,11,12,13,26,27,28,29,30,43,44,45,46,47])=[];
%}

%W([1,9,10,12,20,21,23,31,32],:)=[];
%W(:,[1,9,10,12,20,21,23,31,32])=[];

%W(W(:,:)>1e4)=1e4;
%W(W(:,:)<-1e4)=-1e4;
%W(W(:,:)>1e4)=W(W(:,:)>1e4)/1e3;
%W(W(:,:)<-1e4)=W(W(:,:)<-1e4)/1e3;

% 计算beta
x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) ,  mySample(:,4).^2  , ...,
      mySample(:,6) , mySample(:,7), mySample(:,8), mySample(:,7).*mySample(:,8) ];
y1 = mySample(:,1);
beta1 = (x'*x)\x'*y1;
y2 = mySample(:,2);
beta2 = (x'*x)\x'*y2;
y3 = mySample(:,11);
x3 = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) ,  mySample(:,4).^2  , ...,
    mySample(:,13) , mySample(:,7), mySample(:,8), mySample(:,7).*mySample(:,8), mySample(:,13).*mySample(:,8)];
beta3 = (x3'*x3)\x3'*y3;
beta4 = (x'*x)\x'*y4;

beta1([7,8])=[];
beta2([7,8])=[];
beta3([7,8,10])=[];
%beta4(9)=beta4(9)*(-0.5);
beta4([7,8])=[];
[beta1,beta2,beta3,beta4];
save Sample_did_nosample.mat  W  beta1  beta2  beta3 beta4
save Sample_did.mat W beta1 beta2 beta3 beta4 mySample


%%
%{
% 算一个逐期的出来？
y1 = mySample(:,1);
y2 = mySample(:,2);
y3 = mySample(:,11);
mySample(:,14)=mySample(:,12)==2012;
mySample(:,15)=mySample(:,12)==2014;
mySample(:,16)=mySample(:,12)==2016;
mySample(:,17)=mySample(:,12)==2018;

x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) ,  mySample(:,4).^2  , ...,
     mySample(:,5) , mySample(:,5).^2  , mySample(:,6) , mySample(:,7), mySample(:,14),mySample(:,15), mySample(:,16),mySample(:,17) , ...,
     mySample(:,7).*mySample(:,14),mySample(:,7).*mySample(:,15),mySample(:,7).*mySample(:,16) ,mySample(:,7).*mySample(:,17)];

x3 = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) ,  mySample(:,4).^2  , ...,
     mySample(:,5) , mySample(:,5).^2  , mySample(:,13) , mySample(:,7),  mySample(:,14),mySample(:,15), mySample(:,16),mySample(:,17),...，
     mySample(:,7).*mySample(:,14),mySample(:,7).*mySample(:,15),mySample(:,7).*mySample(:,16) ,mySample(:,7).*mySample(:,17),...,
     mySample(:,13).*mySample(:,14),mySample(:,13).*mySample(:,15),mySample(:,13).*mySample(:,16),mySample(:,13).*mySample(:,17)];

beta1 = (x'*x)\x'*y1;
beta2 = (x'*x)\x'*y2;
beta3 = (x3'*x3)\x3'*y3;
beta1([9,10,11,12,13])=[];
beta2([9,10,11,12,13])=[];
beta3([9,10,11,12,13,18,19,20,21])=[];
save Sample_did_nosample_continue.mat  W  beta1  beta2  beta3
%}




%% did-low fl
fullSample = readmatrix('D:\OneDrive - 北京大学经济学院\房产税问题研究\新工作进展\full_sample.xls');
mySample=fullSample(fullSample(:,9)==0,:);
l=length(mySample);

x_low = x(fullSample(:,9)==0,:);
x3_low= x3(fullSample(:,9)==0,:);

y1_low = mySample(:,1);
y2_low = mySample(:,2);
y3_low = mySample(:,11);
y4_low = mySample(:,10);

beta1 = (x_low'*x_low)\x_low'*y1_low;
beta2 = (x_low'*x_low)\x_low'*y2_low;
beta3 = (x3_low'*x3_low)\x3_low'*y3_low;
beta4 = (x_low'*x_low)\x_low'*y4_low;

beta1([7,8])=[];
beta2([7,8])=[];
beta3([7,8,10])=[];
%beta4(9)=beta4(9)*(-0.5);
beta4([7,8])=[];
[beta1,beta2,beta3,beta4];
save Sample_did_nosample_low.mat  W beta1 beta2  beta3 beta4
save Sample_did_low.mat W beta1 beta2 beta3 beta4 mySample

%%
%{
% 算一个逐期的出来？
y1 = mySample(:,1);
y2 = mySample(:,2);
y3 = mySample(:,11);
mySample(:,14)=mySample(:,12)==2012;
mySample(:,15)=mySample(:,12)==2014;
mySample(:,16)=mySample(:,12)==2016;
mySample(:,17)=mySample(:,12)==2018;

x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) ,  mySample(:,4).^2  , ...,
     mySample(:,5) , mySample(:,5).^2  , mySample(:,6) , mySample(:,7), mySample(:,14),mySample(:,15), mySample(:,16),mySample(:,17) , ...,
     mySample(:,7).*mySample(:,14),mySample(:,7).*mySample(:,15),mySample(:,7).*mySample(:,16) ,mySample(:,7).*mySample(:,17)];

x3 = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) ,  mySample(:,4).^2  , ...,
     mySample(:,5) , mySample(:,5).^2  , mySample(:,13) , mySample(:,7),  mySample(:,14),mySample(:,15), mySample(:,16),mySample(:,17),...，
     mySample(:,7).*mySample(:,14),mySample(:,7).*mySample(:,15),mySample(:,7).*mySample(:,16) ,mySample(:,7).*mySample(:,17),...,
     mySample(:,13).*mySample(:,14),mySample(:,13).*mySample(:,15),mySample(:,13).*mySample(:,16),mySample(:,13).*mySample(:,17)];

beta1 = (x'*x)\x'*y1;
beta2 = (x'*x)\x'*y2;
beta3 = (x3'*x3)\x3'*y3;
beta1([9,10,11,12,13])=[];
beta2([9,10,11,12,13])=[];
beta3([9,10,11,12,13,18,19,20,21])=[];
save Sample_did_nosample_low_continue.mat  W  beta1  beta2  beta3
%}




%% did-high fl
fullSample = readmatrix('D:\OneDrive - 北京大学经济学院\房产税问题研究\新工作进展\full_sample.xls');
mySample=fullSample(fullSample(:,9)==1,:);
l=length(mySample);

x_low = x(fullSample(:,9)==1,:);
x3_low= x3(fullSample(:,9)==1,:);
y1_low = mySample(:,1);
y2_low = mySample(:,2);
y3_low = mySample(:,11);
y4_low = mySample(:,10);

beta1 = (x_low'*x_low)\x_low'*y1_low;
beta2 = (x_low'*x_low)\x_low'*y2_low;
beta3 = (x3_low'*x3_low)\x3_low'*y3_low;
beta4 = (x_low'*x_low)\x_low'*y4_low;

beta1([7,8])=[];
beta2([7,8])=[];
beta3([7,8,10])=[];
%beta4(9)=beta4(9)*(-0.5);
beta4([7,8])=[];
[beta1,beta2,beta3,beta4];

W = W./2;
save Sample_did_nosample_high.mat  W beta1 beta2  beta3 beta4
save Sample_did_high.mat W beta1 beta2 beta3 beta4 mySample

%%
%{
% 算一个逐期的出来？
y1 = mySample(:,1);
y2 = mySample(:,2);
y3 = mySample(:,11);
mySample(:,14)=mySample(:,12)==2012;
mySample(:,15)=mySample(:,12)==2014;
mySample(:,16)=mySample(:,12)==2016;
mySample(:,17)=mySample(:,12)==2018;

x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) ,  mySample(:,4).^2  , ...,
     mySample(:,5) , mySample(:,5).^2  , mySample(:,6) , mySample(:,7), mySample(:,14),mySample(:,15), mySample(:,16),mySample(:,17) , ...,
     mySample(:,7).*mySample(:,14),mySample(:,7).*mySample(:,15),mySample(:,7).*mySample(:,16) ,mySample(:,7).*mySample(:,17)];

x3 = [ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) ,  mySample(:,4).^2  , ...,
     mySample(:,5) , mySample(:,5).^2  , mySample(:,13) , mySample(:,7),  mySample(:,14),mySample(:,15), mySample(:,16),mySample(:,17),...，
     mySample(:,7).*mySample(:,14),mySample(:,7).*mySample(:,15),mySample(:,7).*mySample(:,16) ,mySample(:,7).*mySample(:,17),...,
     mySample(:,13).*mySample(:,14),mySample(:,13).*mySample(:,15),mySample(:,13).*mySample(:,16),mySample(:,13).*mySample(:,17)];

beta1 = (x'*x)\x'*y1;
beta2 = (x'*x)\x'*y2;
beta3 = (x3'*x3)\x3'*y3;
beta1([9,10,11,12,13])=[];
beta2([9,10,11,12,13])=[];
beta3([9,10,11,12,13,18,19,20,21])=[];
save Sample_did_nosample_high_continue.mat  W  beta1  beta2  beta3
%}












%{

%% 实验前+后 prepost
mySample = fullSample;
l = length(mySample);

% 计算权重
x = [ones(l,1) , mySample(:,3) , mySample(:,3).^2 , mySample(:,4) , mySample(:,4).^2 , ...,
     mySample(:,5) , mySample(:,5).^2 , mySample(:,6) ];


k = size(x,2);
lamda1 = zeros(l,k);
lamda2 = zeros(l,k);
qq = zeros(k,k);

for i=1:l
    qq = qq + 1/l * x(i,:)'*x(i,:);
end
Q = inv(qq);

% y1:消费 考虑第一组beta
y1 = mySample(:,1);
beta1 = (x'*x)\x'*y1;

for i=1:l
lamda1(i,:) = (Q*x(i,:)'*(y1(i,:)-x(i,:)*beta1)*sqrt(((l-1)/(l-k))))';
end

% y2:参与 考虑第二组beta
y2 = mySample(:,2);
beta2 = (x'*x)\x'*y2;

for i=1:l
lamda2(i,:) = (Q*x(i,:)'*(y2(i,:)-x(i,:)*beta2)*sqrt(((l-1)/(l-k))))';
end


% 下一组x 来自post %%%%%%%%%%%%%%%%%%%%%%%%%
x1 = [ones(l,1) , mySample(:,3) , mySample(:,3).^2 , mySample(:,4) , mySample(:,4).^2 , ...,
     mySample(:,5) , mySample(:,5).^2 , mySample(:,6)];
k1 = size(x1,2);
lamda3 = zeros(l,k1);
lamda4 = zeros(l,k1);
phiphi = zeros(2*k+2*k1,2*k+2*k1,l);

qq = zeros(k1,k1);
for i=1:l
    qq = qq + 1/l * x1(i,:)'*x1(i,:);
end
Q = inv(qq);

% y1:消费 考虑第一组beta
beta3 = (x1'*x1)\x1'*y1;

for i=1:l
lamda3(i,:) = (Q*x1(i,:)'*(y1(i,:)-x1(i,:)*beta3)*sqrt(((l-1)/(l-k))))';
end

% y2:参与 考虑第二组beta
beta4 = (x1'*x1)\x1'*y2;

for i=1:l
lamda4(i,:) = (Q*x1(i,:)'*(y2(i,:)-x1(i,:)*beta4)*sqrt(((l-1)/(l-k))))';
end




lamda = [lamda1 , lamda2 , lamda3 , lamda4];


for i = 1:l
phiphi(:,:,i) = lamda(i,:)' * lamda(i,:)/l;
end
omega = sum(phiphi,3);

W = inv(omega);
W([9,10],:)=[];
W(:,[9,10])=[];

W3 = W;
save Sample_prepost.mat mySample W x x1 beta1 beta2 beta3 beta4 

%}






