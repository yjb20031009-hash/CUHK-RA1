function [ ggvalue , gvalue , betamat ] = my_estimation_prepostdid1_high_pro(myparam)
%% 【关键外层函数】最外层的函数，也是最优化的目标函数
% 与my_estimation_prepostdid1函数功能完全一致，但仅考虑【高金融素养】样本
%{
clear
myparam=[0.2070    0.00922        0.0628    0.8800    0.4947  0.007  0.5438 0.5507];
[ ggvalue , gvalue , betamat ]=my_estimation_prepostdid1_high_pro(myparam)
%}
%% 预先载入设定好的参数
stept = 2;

tb     = 20/stept; %家户开始的年龄
tr     = 62/stept; %家户退休的年龄
td     = 100/stept; %家户最大的年龄（最后一期）
tn     = td-tb+1; %家户最长的存活时间

ncash  = 11;
nh     = 6; % housing购买量 网格分组

adjcost = 0.07; %住房的调整成本 7%

ret_fac     = 0.6; %退休后的收入因子


maxhouse  = 40.0;
minhouse  = 0.00;  %房产的最小需清偿额度
minhouse2 = 2.8184;  %房产最小可购买门槛

maxcash     = 40.0; 
mincash     = 0.25;

% 实验组对照组 全样本
global incaa incb1 incb2 incb3
% high fl
incb1 = -.1127031;
incb2 = .4746294/100;
incb3 = -.5935427/10000;
incaa = 10.58953;
%global incaa incb1 incb2 incb3

%% 跟fminsearch有关的参数
% ppt         = 0.01; %房产税税率，若为0.5%，则是0.005

%{
ppcost      = 0.025; %股票的per period cost，不应当超过当期收入或退休收入，否则cash会出现负值
otcost      = 0.100; %股票的one time cost
rho         = 10.0; %风险规避系数
delta       = 0.97; %时间贴现因子
psi         = 0.5; %ψ，CES效用函数的跨期替代弹性
%}


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
otcost_exp = myparam(2)*200000/(y60+y61);
otcost_std = myparam(6)*200000/(y60+y61);

    if length(myparam)>6 % 如果input param超过6个

    myparam(7) = myparam(7)*0.20 + 0.00;
    myparam(8) = myparam(8)*0.20 + 0.00;
    
    end

else

ppcost = myparam(1)/(y60+y61);
otcost_exp = myparam(2)/(y60+y61);
otcost_std = myparam(6)/(y60+y61);

end

%% ot cost
% 根据Tauchen-Hussey (1991)将标准正态分布离散为3种情况
[grid,weig2] = tauchenHussey(2,otcost_exp,0,otcost_std,1);
weig_ot = weig2(1,:)';
otcost_mat = [exp(grid+otcost_std^2/2),weig_ot];

%% 区分是实验前还是试验后
% 先导入实验前 得到sim sample
ppt = 0.00;
load mySample_pre10.mat %既包含上海 也包含非上海
mySample=mySample(mySample(:,9)==1,:);


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
% 第一种情况
otcost=otcost_mat(1,1);
if length(myparam)>6
[ C, A, H, C1, A1, H1] = ...,
mymain_se(ppt,ppcost,otcost,rho,delta,psi,myparam(7),myparam(8));
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

if length(myparam)>6
    mu          = myparam(7);
    muh         = myparam(8);
else
    mu          = 0.08 ;
    muh         = 0.08 ;
end

sigr        = 0.42;
sigrh       = 0.28;

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
sim_mySample(:,10) =  0;
sim_mySample(:,11) =  (simAp1(:,:)) * gret_sh(:,3);


sim_mySample_app = [sim_mySample,2010*ones(length(sim_mySample(:,1)),1)];

save sim_mySample2.mat sim_mySample
save sim_mySample_append.mat sim_mySample_app 


%if nargin >= 2 && length(varargin)==2  % 如果期望收益率被参数化了
fake_param = [myparam(1),otcost,myparam(3),myparam(4),myparam(5)];
if length(myparam)>6
% 2012
[ ~ ] = my_estimation_prepost(fake_param,0,myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2012*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2014
[ ~ ] = my_estimation_prepost(fake_param,1,myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2014*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2016
[ ~ ] = my_estimation_prepost(fake_param,1,myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2016*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2018
[ ~ ] = my_estimation_prepost(fake_param,1,myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2018*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app


else
%2012
[ ~ ] = my_estimation_prepost(fake_param,0); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2012*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2014
[ ~ ] = my_estimation_prepost(fake_param,1); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2014*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2016
[ ~ ] = my_estimation_prepost(fake_param,1); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2016*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2018
[ ~ ] = my_estimation_prepost(fake_param,1); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2018*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

end






























%% 求Policy Function
% 第二种情况
otcost=otcost_mat(2,1);
if length(myparam)>6
[ C, A, H, C1, A1, H1] = ...,
mymain_se(ppt,ppcost,otcost,rho,delta,psi,myparam(7),myparam(8));
else
[ C, A, H, C1, A1, H1] = ...,
mymain_se(ppt,ppcost,otcost,rho,delta,psi);
end

% 根据policy function和mySample，计算3*3种情况下的模拟值
n = 3;
nn = n^2;
[grid,weig2] = tauchenHussey(n,0,0,1,1);
weig = weig2(1,:)';

corr_hs     = -0.08; % 股票和房产之间的相关性
r           = 1.05 - 0.048; %无风险利率

if length(myparam)>6
    mu          = myparam(7);
    muh         = myparam(8);
else
    mu          = 0.08 ;
    muh         = 0.08 ;
end

sigr        = 0.42;
sigrh       = 0.28;

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


% Simulation
 
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

% 得到sim sample（第一次）
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
sim_mySample(:,10) =  0;
sim_mySample(:,11) =  (simAp1(:,:)) * gret_sh(:,3);

load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2010*ones(length(sim_mySample(:,1)),1)]]; % sim_mySample 是 新生成的 一年的 数据表
save sim_mySample_append.mat sim_mySample_app % sim_mySample_app 是循环工作的矩阵

fake_param = [myparam(1),otcost,myparam(3),myparam(4),myparam(5)];
if length(myparam)>6 % 如果期望收益率被参数化了
% 2012
[ ~ ] = my_estimation_prepost(fake_param,0,myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2012*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2014
[ ~ ] = my_estimation_prepost(fake_param,1,myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2014*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2016
[ ~ ] = my_estimation_prepost(fake_param,1,myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2016*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2018
[ ~ ] = my_estimation_prepost(fake_param,1,myparam(7),myparam(8)); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2018*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

else
%2012
[ ~ ] = my_estimation_prepost(fake_param,0); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2012*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2014
[ ~ ] = my_estimation_prepost(fake_param,1); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2014*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2016
[ ~ ] = my_estimation_prepost(fake_param,1); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2016*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

%2018
[ ~ ] = my_estimation_prepost(fake_param,1); % 第二个参数 0代表计算policy function 1代表不要重新计算policy function
load sim_mySample2.mat sim_mySample
load sim_mySample_append.mat sim_mySample_app 
sim_mySample_app = [sim_mySample_app ; [sim_mySample,2018*ones(length(sim_mySample(:,1)),1)]];
save sim_mySample_append.mat sim_mySample_app

end











%% 全面混合did
load Sample_did_nosample_high_continue.mat %我们只需要beta和W
load sim_mySample_append.mat
mySample = sim_mySample_app;
l = length(mySample);
mySample(:,13)=mySample(:,12)==2012;
mySample(:,14)=mySample(:,12)==2014;
mySample(:,15)=mySample(:,12)==2016;
mySample(:,16)=mySample(:,12)==2018;

x_sim = [ ones(l,1) , mySample(:,3) , mySample(:,3).^2  , mySample(:,4) , mySample(:,4).^2 , mySample(:,5) , mySample(:,5).^2  , mySample(:,6), mySample(:,7) , ...,
    mySample(:,13),mySample(:,14),mySample(:,15),mySample(:,16),...,
    mySample(:,7).* mySample(:,13),mySample(:,7).* mySample(:,14),mySample(:,7).* mySample(:,15),mySample(:,7).* mySample(:,16),];
y1 = mySample(:,1);
y2 = mySample(:,2);
y3 = mySample(:,11);

% Drop missing value
my_dataframe = [y1,y2,y3,x_sim];
my_dataframe = rmmissing(my_dataframe);
y1 = my_dataframe(:,1);
y2 = my_dataframe(:,2);
y3 = my_dataframe(:,3);
x_sim = my_dataframe(:,4:end);

beta1_sim_mean = (x_sim'*x_sim)\x_sim'*y1;
beta2_sim_mean = (x_sim'*x_sim)\x_sim'*y2;
beta3_sim_mean = (x_sim'*x_sim)\x_sim'*y3;
beta1_sim_mean([9,10,11,12,13])=[];
beta2_sim_mean([9,10,11,12,13])=[];
beta3_sim_mean([9,10,11,12,13])=[];
betamat =  [[beta1;beta2;beta3],[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean]];
gvalue   = [beta1;beta2;beta3]-[beta1_sim_mean;beta2_sim_mean;beta3_sim_mean];

%{
W(17,17)=W(17,17)*500;
W(18,18)=W(18,18)*50000;

W(26,26)=W(26,26)*500;
W(27,27)=W(27,27)*800;
%}
ggvalue  = gvalue'*W*gvalue;




end