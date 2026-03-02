%% 导入数据 筛选25岁以下的样本
clear
data_predict = readmatrix('D:\OneDrive - 北京大学经济学院\房产税问题研究\新工作进展\pictures.xls');
data_predict=sortrows(data_predict,1);

load mySample_pre10.mat %既包含上海 也包含非上海
l=length(mySample(:,1));
mySample=mySample(mySample(:,3)<=35,:);


%% 导入6组参数（全样本，高，低金融素养；参数化期望收益，不参数化期望收益:3*2）
% 设计一个函数，是prepostdid的简化版，以实现以下功能：
% 计算各组参数对应的policy function A(:,:,:) 等 via mymain_se
% 标准+自定义预期收益率
did = [0.1970    0.1183    0.2628    0.9900    0.8947    0.3223 0.3066];
theta   = reshape(did,1,7).*[10000,200000,10,0.29,0.4,0.2,0.2]+[0,0,2,0.7,0.3,0.0,0.0];
tic
[C, A, H, W, C1, A1, H1, W1] = picture_inner_func(0,theta(1),theta(2),theta(3),theta(4),theta(5),theta(6),theta(7));
toc
save Formal_Policyfunc1.mat C  A  H  W  C1  A1  H1  W1

% 标准+默认预期收益率
did = [0.1328    0.2232    0.1775    0.9994    0.8806];
theta   = reshape(did,1,5).*[10000,200000,10,0.29,0.4]+[0,0,2,0.7,0.3];
tic
[C, A, H, W, C1, A1, H1, W1] = picture_inner_func(0,theta(1),theta(2),theta(3),theta(4),theta(5));
toc
save Formal_Policyfunc2.mat C  A  H  W  C1  A1  H1  W1


% HIgh+自定义收益率
didhigh = [0.2070    0.0922        0.0628    0.8800    0.4947    0.5438 0.5507];
theta   = reshape(didhigh,1,7).*[10000,200000,8,0.29,0.4,0.2,0.2]+[0,0,2,0.7,0.3,0.0,0.0];
tic
[C, A, H, W, C1, A1, H1, W1] = picture_inner_func(0,theta(1),theta(2),theta(3),theta(4),theta(5),theta(6),theta(7));
toc
save Formal_Policyfunc3.mat C  A  H  W  C1  A1  H1  W1

% High+默认预期收益率
didhigh = [0.081906  0.079669   0.57396  0.74455  0.58186];
theta   = reshape(didhigh,1,5).*[10000,200000,10,0.29,0.4]+[0,0,2,0.7,0.3];
tic
[C, A, H, W, C1, A1, H1, W1] = picture_inner_func(0,theta(1),theta(2),theta(3),theta(4),theta(5));
toc
save Formal_Policyfunc4.mat C  A  H  W  C1  A1  H1  W1



% Low+自定义收益率
didlow = [0.30040    0.3052    0.0062    0.9987    0.6917    0.3326    0.3462];
theta   = reshape(didlow,1,7).*[10000,200000,8,0.29,0.4,0.2,0.2]+[0,0,2,0.7,0.3,0.0,0.0];
tic
[C, A, H, W, C1, A1, H1, W1] = picture_inner_func(0,theta(1),theta(2),theta(3),theta(4),theta(5),theta(6),theta(7));
toc
save Formal_Policyfunc5.mat C  A  H  W  C1  A1  H1  W1

% Low+默认预期收益率
didlow = [0.2523    0.3904    0.0111    0.9684    0.9820];
theta   = reshape(didlow,1,5).*[10000,200000,10,0.29,0.4]+[0,0,2,0.7,0.3];
tic
[C, A, H, W, C1, A1, H1, W1] = picture_inner_func(0,theta(1),theta(2),theta(3),theta(4),theta(5));
toc
save Formal_Policyfunc6.mat C  A  H  W  C1  A1  H1  W1



%%
global incaa incb1 incb2 incb3
incb1 = .0427089;
incb2 = -.0216383/100;
incb3 = -.1124394/10000;
incaa = 8.191964;
% B 格点
global ncash nh
ncash  = 41; % 状态变量cash格点
nh     = 21; % 状态变量housing格点

%{


load Formal_Policyfunc1.mat

figure(103)
subplot(2,2,1)
mesh(ghouse,gcash,A(:,:,3))
subplot(2,2,2)
mesh(ghouse,gcash,H(:,:,3))
subplot(2,2,3)
mesh(ghouse,gcash,A1(:,:,3))
subplot(2,2,4)
mesh(ghouse,gcash,H1(:,:,3))

figure(110)
subplot(2,2,1)
mesh(ghouse,gcash,A(:,:,10))
subplot(2,2,2)
mesh(ghouse,gcash,H(:,:,10))
subplot(2,2,3)
mesh(ghouse,gcash,A1(:,:,10))
subplot(2,2,4)
mesh(ghouse,gcash,H1(:,:,10))

figure(120)
subplot(2,2,1)
mesh(ghouse,gcash,A(:,:,20))
subplot(2,2,2)
mesh(ghouse,gcash,H(:,:,20))
subplot(2,2,3)
mesh(ghouse,gcash,A1(:,:,20))
subplot(2,2,4)
mesh(ghouse,gcash,H1(:,:,20))

%}




%% 使用大循环，计算每个人、每一期的wealth和A
method='nearest';
for nameid=1:6
    filename = strcat('Formal_Policyfunc',num2str(nameid),'.mat');
    load(filename)
    data_predict = readmatrix('D:\OneDrive - 北京大学经济学院\房产税问题研究\新工作进展\pictures.xls');
    %两列交换顺序
    data_predict(:,8:9)=data_predict(:,4:5);
    data_predict(:,3:4)=[];
    data_predict=sortrows(data_predict,1);
    data_predict=data_predict(:,[1,ceil(nameid/2)*2,ceil(nameid/2)*2+1]);
    load mySample_pre10.mat %既包含上海 也包含非上海
    l=length(mySample(:,1));

%%
% 1+2
if ceil(nameid/2)==1
mySample=mySample(mySample(:,3)<=90,:);
% 3+4
elseif ceil(nameid/2)==2
mySample=mySample(mySample(:,3)<=90&mySample(:,9)==1,:);
% 5+6
elseif ceil(nameid/2)==3
mySample=mySample(mySample(:,3)<=90&mySample(:,9)==0,:);
end

%% 统一程序
stept = 2;
tb     = 20/stept; %家户开始的年龄
tr     = 62/stept; %家户退休的年龄
td     = 100/stept; %家户最大的年龄（最后一期）
tn     = td-tb+1; %家户最长的存活时间
maxhouse  = 40.0;
minhouse  = 0.00;  %房产的最小需清偿额度
minhouse2 = 2.8184;  %房产最小可购买门槛
maxcash     = 40.0; 
mincash     = 0.25;
global ncash nh
l_maxcash = log(maxcash);
l_mincash = log(mincash);
stepcash = (l_maxcash-l_mincash)/(ncash-1); %最大最小财富之差除以50
l_maxhouse = log(maxhouse+1);
l_minhouse = log(minhouse+1);
stephouse = (l_maxhouse-l_minhouse)/(nh-1); %最大最小房产之差除以50
for i1=1:ncash
   lgcash(i1,1)=l_mincash+(i1-1.0)*stepcash;  %把财富按ln切割成50个区间，lgcash是50*1
end
for i1=1:ncash
   gcash(i1,1)=exp(lgcash(i1,1)); %对lgcash取指数，50*1
end
for i1=1:nh
   lghouse(i1,1)=l_minhouse+(i1-1.0)*stephouse;   
end
for i1=1:nh
   ghouse(i1,1)=exp(lghouse(i1,1))-1;  
end

mySample(:,13)=mySample(:,5)+mySample(:,4);
for t=1:tn-1
    %% 插值计算
    age = 20+t*2;
    Part_sample    = mySample(mySample(:,6)==1&max(20,floor(mySample(:,3)/2)*2)==age,:);
    nonPart_sample = mySample(mySample(:,6)==0&max(20,floor(mySample(:,3)/2)*2)==age,:);
    l_temp1 = length(Part_sample(:,1));
    l_temp2 = length(nonPart_sample(:,1));
    Part_sim_W=NaN(l_temp1,1);
    Part_sim_A=NaN(l_temp1,1);
    Part_sim_H=NaN(l_temp1,1);
    Part_sim_Cash=NaN(l_temp1,1);
    nonPart_sim_W=NaN(l_temp2,1);
    nonPart_sim_A=NaN(l_temp2,1);
    nonPart_sim_H=NaN(l_temp2,1);
    nonPart_sim_Cash=NaN(l_temp2,1);

    Part_sim_W(:,1)    = interp2(ghouse,gcash,W(:,:,t),Part_sample(:,5),max(0.25,Part_sample(:,4)),method);
    nonPart_sim_W(:,1) = interp2(ghouse,gcash,W1(:,:,t),nonPart_sample(:,5),max(0.25,nonPart_sample(:,4)),method);

    Part_sim_A(:,1)    = interp2(ghouse,gcash,A(:,:,t),Part_sample(:,5),max(0.25,Part_sample(:,4)),method);
    nonPart_sim_A(:,1) = interp2(ghouse,gcash,A1(:,:,t),nonPart_sample(:,5),max(0.25,nonPart_sample(:,4)),method);

    Part_sim_H(:,1)    = interp2(ghouse,gcash,H(:,:,t),Part_sample(:,5),max(0.25,Part_sample(:,4)),method);
    nonPart_sim_H(:,1) = interp2(ghouse,gcash,H1(:,:,t),nonPart_sample(:,5),max(0.25,nonPart_sample(:,4)),method);

    Part_sim_Cash(:,1)    = max(0 , Part_sim_W(:,1) - Part_sim_H(:,1));
    nonPart_sim_Cash(:,1) = max(0, nonPart_sim_W(:,1) - nonPart_sim_H(:,1));
    
    l_temp = length(Part_sample(:,1)) + length(nonPart_sample(:,1));
    %% 迭代样本
    new_Sample = NaN(l_temp,13);
    new_Sample(:,3) = age+2;
    new_Sample(:,4) = [Part_sim_Cash;nonPart_sim_Cash];
    new_Sample(:,5) = [Part_sim_H;nonPart_sim_H];
    new_Sample(:,6) = [Part_sim_A>0;nonPart_sim_A>0];
    new_Sample(:,13) = [Part_sim_W;nonPart_sim_W];

    mySample = [mySample;new_Sample];

end


mySample(:,14)=floor(mySample(:,3)/2)*2;
W2PI = NaN((max(mySample(:,14))-min(mySample(:,14)))/2,2);

y_temp = mySample(:,13);
x_temp = [mySample(:,14),mySample(:,14).^2,mySample(:,14).^3];
b_temp = (x_temp'*x_temp)^(-1)*x_temp'*y_temp;
%y_hat_temp = x_temp*b_temp;

for i=1:(max(mySample(:,14))-min(mySample(:,14)))/2
    W2PI(i,1)=i*2+min(mySample(:,14))-2;
    %W2PI(i,2)=mean(mySample(mySample(:,14)==W2PI(i,1),13));
    W2PI(i,2)=[W2PI(i,1),W2PI(i,1)^2,W2PI(i,1)^3]*b_temp;
end

y_temp = mySample(:,6);
b_temp = (x_temp'*x_temp)^(-1)*x_temp'*y_temp;

AA = NaN((max(mySample(:,14))-min(mySample(:,14)))/2,2);
for i=1:(max(mySample(:,14))-min(mySample(:,14)))/2
    AA(i,1)=i*2+min(mySample(:,14))-2;
    %AA(i,2)=mean(mySample(mySample(:,14)==W2PI(i,1),6));
    AA(i,2)=[W2PI(i,1),W2PI(i,1)^2,W2PI(i,1)^3]*b_temp;
end

%%
fontsize = 20;
figure(ceil(nameid/2))
%% 1 3 5
if mod(nameid,2)==1
subplot(2,2,1)
plot(W2PI(1:31,1),W2PI(1:31,2),'LineWidth',3),xlabel("Age",'FontSize',fontsize), ylabel("Wealth/PI",'FontSize',fontsize)
hold on
plot(data_predict(1:63,1),data_predict(1:63,2),'--','LineWidth',3),legend('Model','Data','FontSize',fontsize)
axis([30 80 0 5])
set(gca,'FontSize',fontsize)

subplot(2,2,2)
plot(AA(1:31,1),AA(1:31,2),'LineWidth',3),xlabel("Age",'FontSize',fontsize), ylabel("Participant",'FontSize',fontsize)
hold on
plot(data_predict(1:63,1),data_predict(1:63,3),'--','LineWidth',3),legend('Model','Data','FontSize',fontsize)
axis([30 80 0 1])
set(gca,'FontSize',fontsize)
end

%% 2 4 6
if mod(nameid,2)==0
subplot(2,2,3)
plot(W2PI(1:31,1),W2PI(1:31,2),'LineWidth',3),xlabel("Age",'FontSize',fontsize), ylabel("Wealth/PI",'FontSize',fontsize)
hold on
plot(data_predict(1:63,1),data_predict(1:63,2),'--','LineWidth',3),legend('Model','Data','FontSize',fontsize)
axis([30 80 0 5])
set(gca,'FontSize',fontsize)

subplot(2,2,4)
plot(AA(1:31,1),AA(1:31,2),'LineWidth',3),xlabel("Age",'FontSize',fontsize), ylabel("Participant",'FontSize',fontsize)
hold on
plot(data_predict(1:63,1),data_predict(1:63,3),'--','LineWidth',3),legend('Model','Data','FontSize',fontsize)
axis([30 80 0 1])
set(gca,'FontSize',fontsize)
end

end

