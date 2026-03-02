function [ ggvalue , gvalue , betamat ] = my_estimation_prepostdid1_low(myparam)
%% 【关键外层函数】最外层的函数，也是最优化的目标函数
% 与my_estimation_prepostdid1函数功能完全一致，但仅考虑【低金融素养】样本

%% 预先载入设定好的参数
stept = 2;

tb     = 20/stept; %家户开始的年龄
tr     = 62/stept; %家户退休的年龄
td     = 100/stept; %家户最大的年龄（最后一期）
tn     = td-tb+1; %家户最长的存活时间

global ncash nh incaa incb1 incb2 incb3
ncash  = 21;
nh     = 11; % housing购买量 网格分组

adjcost = 0.07; %住房的调整成本 7%

ret_fac     = 0.6; %退休后的收入因子


maxhouse  = 40.0;
minhouse  = 0.00;  %房产的最小需清偿额度
minhouse2 = 2.5563*0;  %房产最小可购买门槛

maxcash     = 40.0; 
mincash     = 0.25;

% 实验组对照组 全样本
% low fl
incb1 = .0096951;
incb2 =  -1.81387/1e4;
incb3 = 0/1e6;
incaa =  9.87492;
%global incaa incb1 incb2 incb3



%% 跟fminsearch有关的参数

if myparam(2)<1 %说明是结构参数

rho    = myparam(3)*10+2;
delta  = myparam(4)*0.29+0.70;
psi    = myparam(5)*0.40+0.30;

else

rho    = myparam(3);
delta  = myparam(4);
psi    = myparam(5);

end


y60 = exp( incaa + incb1*60 + incb2*60^2 + incb3*60^3 );
y61 = exp( incaa + incb1*61 + incb2*61^2 + incb3*61^3 );


if myparam(2)<1 %说明是结构参数


ppcost = myparam(1)*10000/(y60+y61);
otcost = myparam(2)*200000/(y60+y61);

    %if nargin >= 2 && length(varargin)>=2  % 如果期望收益率被参数化了
    if length(myparam)>5 % 如果input param超过五个

    %varargin{length(varargin)-1} = varargin{length(varargin)-1}*0.40 + 0.00;
    myparam(6) = myparam(6)*0.20 + 0.00;
    %varargin{length(varargin)} = varargin{length(varargin)}*0.40 + 0.00;
    myparam(7) = myparam(7)*0.20 + 0.00;
    
    end

    if length(myparam)>7
    myparam(8) = myparam(8)*0.20 + 0.00;
    end

else

ppcost = myparam(1)/(y60+y61);
otcost = myparam(2)/(y60+y61);

end






%% 区分是实验前还是试验后
% 先导入实验前 得到sim sample
ppt = 0.00;
load mySample_pre10.mat %既包含上海 也包含非上海
mySample=mySample(mySample(:,9)==0,:);

%{
if nargin>1 % 调节高低金融素养 1是低金融素养 2是高金融素养 

    if varargin{1}==1
        mySample=mySample(mySample(:,9)==0,:);
    elseif varargin{1}==2
        mySample=mySample(mySample(:,9)==1,:);
    end

end
%}

%% 打格子（for 状态变量）
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

%% 求Policy Function
%if nargin >= 2 && length(varargin)==2  % 如果期望收益率被参数化了
if length(myparam)>5
[ C, A, H, C1, A1, H1] = ...,
mymain_se(ppt,ppcost,otcost,rho,delta,psi,myparam(6),myparam(7));
%mymain_se(ppt,ppcost,otcost,rho,delta,psi,varargin{length(varargin)-1},varargin{length(varargin)});
else
[ C, A, H, C1, A1, H1] = ...,
mymain_se(ppt,ppcost,otcost,rho,delta,psi);
end
 

%% 根据policy function和mySample，计算3*3种情况下的模拟值
n = 3;
nn = n^2;
[grid,weig2] = tauchenHussey(n,0,0,1,1);
weig = weig2(1,:)';

corr_hs     = -0.08; % 股票和房产之间的相关性

r           = 1.05 - 0.048; %无风险利率

%if nargin >= 2 && length(varargin)==2 %如果声明了两类资产的回报预期
if length(myparam)>5
    mu          = myparam(6);%varargin{length(varargin)-1}; %股票超额回报
    muh         = myparam(7);%varargin{length(varargin)};
else
    mu          = 0.08 ; %股票超额回报
    muh         = 0.08 ;
end

sigr        = 0.42; %股票波动率
sigrh       = 0.28; %房产波动率

gret        = zeros(n,1);
greth        = zeros(n,n);
gret_sh      = zeros(nn,3);

for i1=1:n
    gret(i1,1) = r+mu+grid(i1,1)*sigr; %模拟的风险资产回报
end

for i1=1:n
    grid2(:,1) = grid(i1,1)*corr_hs+grid(:,1).*ones(n,1)*(1-corr_hs^2)^(0.5);
    greth(:,i1) = r+muh+grid2(:,1)*sigrh; %模拟的房产回报
end
greth = reshape(greth,nn,1);

for i1=1:nn
    gret_sh(i1,1)=gret(ceil((i1)/n),1);
    gret_sh(i1,2)=greth(i1);
    gret_sh(i1,3)=weig(ceil((i1)/n))*weig(mod(i1-1,n)+1);
end


%% Simulation
 
l = length(mySample);

gyp = zeros(tn,1);

for t=1:tn
if t>=tr-tb+1
    gyp(t,1) = 1;
else
    f_y1   = exp(incaa+incb1*(stept*(t+tb-1))+incb2*(stept*(t+tb-1))^2+incb3*(stept*(t+tb-1))^3);
    f_y1_2 = exp(incaa+incb1*(stept*(t+tb-1)-1)+incb2*(stept*(t+tb-1)-1)^2+incb3*(stept*(t+tb-1)-1)^3);
    f_y2   = exp(incaa+incb1*(stept*(t+tb))+incb2*(stept*(t+tb))^2+incb3*(stept*(t+tb))^3);
    f_y2_2 = exp(incaa+incb1*(stept*(t+tb)-1)+incb2*(stept*(t+tb)-1)^2+incb3*(stept*(t+tb)-1)^3);

    gyp(t,1)  = exp((f_y2+f_y2_2)/(f_y1+f_y1_2)-1.0);
end
end


otcost_t = zeros(tn,1);
ppcost_t = zeros(tn,1);
minhouse2_t = zeros(tn,1);

otcost_t(tn,1) = otcost;
ppcost_t(tn,1) = ppcost;
minhouse2_t(tn,1) = minhouse2;

for t=tn-1:-1:1
otcost_t(t,1) = otcost_t(t+1,1)*gyp(t);
ppcost_t(t,1) = ppcost_t(t+1,1)*gyp(t);
minhouse2_t(t,1)= minhouse2_t(t+1)*gyp(t);
end




% 设置初值
initialW     = mySample(:,4);
initialH     = mySample(:,5);
initialAge   = floor((mySample(:,3)-20)/2)+1;
initialAge(find(initialAge<1),1) = 1;
initialIpart = mySample(:,6);

simC = zeros(l,1);
simA = zeros(l,1);
simH = zeros(l,1);
simS = zeros(l,1);

simW      = zeros(l,9);
simH2     = zeros(l,9);
simAge    = initialAge+1;
simI      = zeros(l,1);

simAp1   = zeros(l,9);
simCp1   = zeros(l,9);



for i = 1:l
t = initialAge(i,1);

        if t > tr-tb
       cash  = initialW(i,1) + ret_fac;
        else
       cash  = initialW(i,1) + 1;
        end
       house = initialH(i,1);

        if initialIpart(i,1) == 0
       simC(i,1) = interp2(ghouse,gcash,C1(:,:,t),house,cash);
       simA(i,1) = interp2(ghouse,gcash,A1(:,:,t),house,cash);
       simH(i,1) = interp2(ghouse,gcash,H1(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A1(:,:,t),house,cash,"nearest");
       simHa = interp2(ghouse,gcash,H1(:,:,t),house,cash,"nearest");

       simA(i,1)= (1 - (simAa==0)) *  simA(i,1);
       simH(i,1)= (1 - (simHa==0)) *  simH(i,1);

        else
       simC(i,1) = interp2(ghouse,gcash,C(:,:,t),house,cash);
       simA(i,1) = interp2(ghouse,gcash,A(:,:,t),house,cash);
       simH(i,1) = interp2(ghouse,gcash,H(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A(:,:,t),house,cash,"nearest");
       simHa = interp2(ghouse,gcash,H(:,:,t),house,cash,"nearest");

       simA(i,1)= (1 - (simAa==0)) *  simA(i,1);
       simH(i,1)= (1 - (simHa==0)) *  simH(i,1);
        end

        %如果Ipart=0且当期参加了股票，则取1；如果Ipart=1，则也取1
        if (initialIpart(i,1) == 0 && simA(i,1)>0) || initialIpart(i,1) == 1
            simI(i,1) = 1;
        end

        if abs((simH(i,1)-house)) <= 0.05*house || (house==0.0 && simH(i,1)<minhouse2_t(t,1)*0.9)
            simH(i,1) = house;
        end

        if house==0.0 && simH(i,1)<minhouse2_t(t,1) && simH(i,1)>=minhouse2_t(t,1)*0.9
            simH(i,1) = minhouse2_t(t,1);
        end



        con1 = simH(i,1) == house ;
        con2 = simI(i,1)-initialIpart(i,1) == 1;
        con3 = simA(i,1)>0;

    simS(i,1) = cash + house*(1-ppt-(1-con1)*adjcost)-simC(i,1)-simH(i,1) - con2*otcost_t(t,1) - con3.*ppcost_t(t,1);



    simW(i,:)  = (simS(i,1)*simA(i,1)*gret_sh(:,1)'+simS(i,1)*(1-simA(i,1))*r*ones(1,nn))/gyp(t,1);
    simH2(i,:) = simH(i,1)*gret_sh(:,2)'/gyp(t,1);

% 接下来用T+1的状态变量代入Policy Function，求解T+1期的Choice
t =  simAge(i,1);

        if t > tr-tb
       cash  = simW(i,:) + ret_fac;
        else
       cash  = simW(i,:) + 1;
        end
       house = simH2(i,:);

        if simI(i,1) == 0
       simCp1(i,:) = interp2(ghouse,gcash,C1(:,:,t),house,cash);
       simAp1(i,:) = interp2(ghouse,gcash,A1(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A1(:,:,t),house,cash,"nearest");
       simAp1(i,simAa==0) = 0;

        else
       simCp1(i,:) = interp2(ghouse,gcash,C(:,:,t),house,cash);
       simAp1(i,:) = interp2(ghouse,gcash,A(:,:,t),house,cash);

       simAa = interp2(ghouse,gcash,A(:,:,t),house,cash,"nearest");
       simAp1(i,simAa==0) = 0;

        end

end





%% 得到sim sample（第一次）
sim_mySample = [];
sim_mySample(:,1) = simCp1(:,:) * gret_sh(:,3);
sim_mySample(:,2) = (simAp1(:,:)>0) * gret_sh(:,3);
sim_mySample(:,3) = (mySample(:,3)+2);
sim_mySample(:,4) =   simW(:,:) * gret_sh(:,3);
sim_mySample(:,5) =  simH2(:,:) * gret_sh(:,3);
sim_mySample(:,6) =  simI(:,1);
sim_mySample(:,7) =  (mySample(:,7));
sim_mySample(:,8) =  (mySample(:,8));
sim_mySample(:,9) =  (mySample(:,9));
sim_mySample(:,10) =  sim_mySample(:,4)./(sim_mySample(:,4)+sim_mySample(:,5)); % fin ratio
sim_mySample(:,11) =  (simAp1(:,:)) * gret_sh(:,3);

sim_mySample_app = sim_mySample;

save sim_mySample2.mat sim_mySample
save sim_mySample_append.mat sim_mySample_app 


%if nargin >= 2 && length(varargin)==2  % 如果期望收益率被参数化了
if length(myparam)>5
% 2012
[ ~ ] = my_estimation_prepost([ppcost,otcost,rho,delta,psi],0,myparam(6),myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; sim_mySample];
save sim_mySample_append.mat sim_mySample_app

%2014
[ ~ ] = my_estimation_prepost([ppcost,otcost,rho,delta,psi],1,myparam(6),myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; sim_mySample];
save sim_mySample_append.mat sim_mySample_app

%2016
[ ~ ] = my_estimation_prepost([ppcost,otcost,rho,delta,psi],1,myparam(6),myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; sim_mySample];
save sim_mySample_append.mat sim_mySample_app

%2018
[ ~ ] = my_estimation_prepost([ppcost,otcost,rho,delta,psi],1,myparam(6),myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; sim_mySample];
save sim_mySample_append.mat sim_mySample_app


else
%2012
[ ~ ] = my_estimation_prepost([ppcost,otcost,rho,delta,psi],0); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; sim_mySample];
save sim_mySample_append.mat sim_mySample_app

%2014
[ ~ ] = my_estimation_prepost([ppcost,otcost,rho,delta,psi],1); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; sim_mySample];
save sim_mySample_append.mat sim_mySample_app

%2016
[ ~ ] = my_estimation_prepost([ppcost,otcost,rho,delta,psi],1); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; sim_mySample];
save sim_mySample_append.mat sim_mySample_app

%2018
[ ~ ] = my_estimation_prepost([ppcost,otcost,rho,delta,psi],1); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; sim_mySample];
save sim_mySample_append.mat sim_mySample_app

end


% 全面混合did
load Sample_did_nosample_low.mat %我们只需要beta和W
load sim_mySample_append.mat
mySample = sim_mySample_app;
l = length(mySample);
x_sim = [ ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) , mySample(:,4).^2 , mySample(:,5) , mySample(:,5).^2  , mySample(:,6), mySample(:,7) , mySample(:,8) ,mySample(:,7).*mySample(:,8)];
x_sim4 = [ ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) , mySample(:,4).^2 , mySample(:,6), mySample(:,7) , mySample(:,8) ,mySample(:,7).*mySample(:,8)];

y1 = mySample(:,1);
y2 = mySample(:,2);
y3 = mySample(:,11);
y4 = mySample(:,10);% fin ratio

% Drop missing value
my_dataframe = [y1,y2,y3,y4,x_sim,x_sim4];
my_dataframe = rmmissing(my_dataframe);
y1 = my_dataframe(:,1);
y2 = my_dataframe(:,2);
y3 = my_dataframe(:,3);
y4 = my_dataframe(:,4);
x_sim = my_dataframe(:,5:15);
x_sim4= my_dataframe(:,16:24);


beta1_sim_mean = (x_sim'*x_sim)\x_sim'*y1;
beta2_sim_mean = (x_sim'*x_sim)\x_sim'*y2;
beta3_sim_mean = (x_sim'*x_sim)\x_sim'*y3;
beta4_sim_mean = (x_sim4'*x_sim4)\x_sim4'*y4;

% Remove the two-way fixed effects
beta1_sim_mean([9,10])=[];
beta2_sim_mean([9,10])=[];
beta3_sim_mean([9,10])=[];
%beta4_sim_mean([9,10])=[];
%beta1_sim_mean([7,8])=[];
%beta2_sim_mean([7,8])=[];
%beta3_sim_mean([7,8])=[];
beta4_sim_mean([7,8])=[];
%{
W(17,17)=W(17,17)*20;
W(18,18)=W(18,18)*500;

W(26,26)=W(26,26)*50;
W(27,27)=W(27,27)*500;

W(33,33)=W(33,33)*5;
W(34,34)=W(34,34)*2000;
%}
%{
W(13,13)=W(13,13)*20;
W(14,14)=W(14,14)*100;

W(20,20)=W(20,20)*50;
W(21,21)=W(21,21)*200;

W(27,27)=W(27,27)*2;
W(28,28)=W(28,28)*200;
%}

betamat =  [[beta1;beta2;beta3;beta4],[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean;beta4_sim_mean]]; % empirical and model-implied moments
gvalue   = [beta1;beta2;beta3;beta4]-[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean;beta4_sim_mean]; % differences between them


ggvalue  = gvalue'*W*gvalue;




end