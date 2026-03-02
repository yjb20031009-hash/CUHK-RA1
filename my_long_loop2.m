function [cash_1,house_1,gc_1d,ghouse2_1d,u,otcost,ppcost,minhouse2] = my_long_loop2(t,Param1,Param2,Param3,grid,weig,otcost,ppcost,minhouse2)
%% 重要：本函数是内置函数，不需要单独使用，也没有关键参数

%% Life-Cycle Consumption/Portfolio Choice Problem
% 本函数是mymain_se函数的内置函数，不能脱离mymain_se单独使用
% 这个函数与my_long_loop3功能相似，但这里包含股票这类资产：
% 给出一组参数，求解"每一组状态变量下、每一组choice"对应的Value funtion的取值
% 得到上述结果后，回到mymain_se中，选出每一组状态变量下的最大值，进而得到policy funtion
% cash_1,house_1,gc_1d,ghouse2_1d 是一系列长向量，用于记录不同的state和choice
% u是一个长向量，代表这组选择下的当期效用
% otcost,ppcost,minhouse2是为了技术上的方便，没有实际含义

%% Variable Definitions

stept = Param3(13);

tb     = Param1(1); %家户开始的年龄
tr     = Param1(2); %家户退休的年龄
%td     = Param1(3); %家户最大的年龄（最后一期）
tn     = Param1(4); %家户最长的存活时间

na     = Param1(5); 
ncash  = Param1(6);
n      = Param1(7);
nc     = Param1(8);

nn     = n*n;


% Mark
nh = Param1(9); % housing购买量 网格分组
nh2 = Param3(12); % housing购买量 网格分组

adjcost = Param1(10); %住房的调整成本
%ppcost = Param1(11); %股票的per period cost
%otcost = Param1(12); %股票的one time cost

minalpha = Param1(13); %股票的最小投资比例（如果选择pay participation cost的话）

maxhouse  = Param1(14);
minhouse  = Param2(1);  %房产的最小可清偿额度
%minhouse2 = Param2(2);  %房产最小可购买门槛


maxcash     = Param2(3); 
mincash     = Param2(4);
aa          = Param2(5); %收入对年龄回归 常数项
b1          = Param2(6); %收入对年龄的系数
b2          = Param2(7); 
b3          = Param2(8); 
ret_fac     = Param2(9); %？退休后的收入因子
%smay        = Param2(10); 
%smav        = Param2(11); 

corr_hs     = Param2(12);

%corr_v      = Param3(1);  %劳动收入的风险与股票风险的相关性？
%corr_y      = Param3(2);  %劳动收入的风险与股票风险的相关性？
%rho         = Param3(3); %风险规避系数
delta       = Param3(4); %时间贴现因子
psi         = Param3(5); %ψ，CES效用函数的跨期替代弹性
r           = Param3(6); %无风险利率
mu          = Param3(7); %股票超额回报
muh         = Param3(8);
sigr        = Param3(9); %波动率
sigrh       = Param3(10); %房产波动率

ppt         = Param3(11); %房产税税率



%survprob    = zeros(tn-1,1);
%delta2      = zeros(tn-1,1);
%grid        = zeros(n,1);
%weig        = zeros(n,1);
gret        = zeros(n,1);
% MARK
greth        = zeros(n,n);
gret_sh      = zeros(nn,3);




ones_n_1    = ones(n,1);
grid2       = zeros(n,1);
%yp          = zeros(n,n);
%yh          = zeros(n,n);
%nweig3      = zeros(n,n,n);
%nweig2      = zeros(n,n);
%nweig4      = zeros(n,n,n,n);
%f_y         = zeros(tr-tb+1,1);
%gy          = zeros(tr-tb,1);
%gyp         = zeros(n,n,tn-1); %标准化的收入，在不同风险因子下
gcash       = zeros(ncash,1); %加入housing
lgcash      = zeros(ncash,1); %加入housing

%Mark
%H           = ones(ncash,nh,tn); %不同Cash下 下一期housing的决策
ghouse       = zeros(nh,1); %加入housing
lghouse      = zeros(nh,1); %加入housing


ga          = zeros(na,1);
riskret     = zeros(na,nn);

%% Additional Computations


for i1=1:n
    gret(i1,1) = r+mu+grid(i1,1)*sigr; %模拟的风险资产回报
end

for i1=1:n
    grid2(:,1) = grid(i1,1)*corr_hs+grid(:,1).*ones_n_1(:,1)*(1-corr_hs^2)^(0.5);
    greth(:,i1) = r+muh+grid2(:,1)*sigrh; %模拟的房产回报
end
greth = reshape(greth,nn,1);

for i1=1:nn
    gret_sh(i1,1)=gret(ceil((i1)/n),1);
    gret_sh(i1,2)=greth(i1);
    gret_sh(i1,3)=weig(ceil((i1)/n))*weig(mod(i1-1,n)+1);
end
% gret_sh 三列 分别是股票收益率 房产收益率 发生的概率权重


%theta = (1.0-rho)/(1.0-1.0/psi); 
psi_1 = 1.0-1.0/psi;
%psi_2 = 1.0/psi_1;

%% Grids for the State Variables and for Portfolio Rule

for i1=1:na
   ga(i1,1)= minalpha+(i1-1)*(1-minalpha)/(na-1); 
   % ga向量：1, 49/50, 48/50......, 0.05，模拟股票的比例alpha，最小值为5%
end


for i5=1:na
   for i8=1:nn
      riskret(i5,i8)=r*(1-ga(i5,1))+gret_sh(i8,1)*ga(i5,1); 
      %riskret矩阵51*5，无风险利率*比例+股票收益率（5个可能）*（1-比例）
      %不包含不买股票的情况
   end
end

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
% ghouse(1,1) = 0;
% 以上对cash和house都打了网格


%% Labor Income

if t>=tr-tb
    gyp = 1;
else
    f_y1   = exp(aa+b1*(stept*(t+tb+1))+b2*(stept*(t+tb+1))^2+b3*(stept*(t+tb+1))^3);
    f_y1_2 = exp(aa+b1*(stept*(t+tb+1)+1)+b2*(stept*(t+tb+1)+1)^2+b3*(stept*(t+tb+1)+1)^3);
    f_y2   = exp(aa+b1*(stept*(t+tb))+b2*(stept*(t+tb))^2+b3*(stept*(t+tb))^3);
    f_y2_2 = exp(aa+b1*(stept*(t+tb)+1)+b2*(stept*(t+tb)+1)^2+b3*(stept*(t+tb)+1)^3);

    gyp  = exp((f_y1+f_y1_2)/(f_y2+f_y2_2)-1.0);
end


otcost = otcost*gyp;
ppcost = ppcost*gyp;
minhouse2 = minhouse2*gyp;

%% Retirement Periods 

i1=1:nh;
i3=1:ncash;
i4=1:nc;
i5=1:na;
i2=1:nh2;
i11=1:nn;

[i11_nn,i5_na,i2_nh,i4_nc,i3_ncash,i1_nh]=ndgrid(i11,i5,i2,i4,i3,i1);


%VV_temp = V(:,:,t+1);
gc_1d = zeros(numel(i4_nc),1);
ghouse2_1d = zeros(numel(i4_nc),1);

cash_1 = zeros(numel(i4_nc),1);
house_1 = zeros(numel(i4_nc),1);
%sav =  zeros(numel(i4_nc),1);
u = zeros(numel(i4_nc),1);
gc = zeros(nc,numel(i4_nc));
ghouse2 =zeros(nh2,numel(i4_nc));



for ii=1:nn*na*nh2*nc:numel(i4_nc)
    minc = 0.25;
    maxc = gcash(i3_ncash(ii))+ghouse(i1_nh(ii))*(1-adjcost-ppt);
    gc(:,ii:ii+nn*na*nh2*nc-1) = linspace(minc,maxc,nc)'*ones(1,nn*na*nh2*nc);
end





for ii=1:nn*na*nh2:numel(i4_nc)

u(ii:ii+nn*na*nh2-1,1)=(1.0-delta)*(gc(i4_nc(ii),ii)^psi_1) ; %每种消费下的utility
gc_1d(ii:ii+nn*na*nh2-1,1)       =  gc(i4_nc(ii),ii);

minh = minhouse2;

if  gc(i4_nc(ii),ii) + ghouse(i1_nh(ii))*ppt > gcash(i3_ncash(ii)) %假如消费+房产税比cash要多，则肯定需要处置房产
maxh =  ghouse(i1_nh(ii))*(1-adjcost-ppt) + gcash(i3_ncash(ii)) - gc(i4_nc(ii),ii);
    if maxh < minh
    ghouse2(:,ii:ii+nn*na*nh2-1) = zeros(nh2,1)*ones(1,nn*na*nh2); %则没法持有房产
    else
    ghouse2(:,ii:ii+nn*na*nh2-1) =  [0;exp(linspace(log(minh),log(maxh),nh2-1))']*ones(1,nn*na*nh2);
    end


 % 此时cash扣掉消费 大于 调整成本 也就是说最多买ghouse(i1_nh(ii))*(1-adjcost) + gcash(i3_ncash(ii)) - gc(i4_nc(ii))


else %可处置可不处置

maxh = min( ghouse(i1_nh(ii))*(1-adjcost-ppt) + gcash(i3_ncash(ii)) - gc(i4_nc(ii),ii) , maxhouse );
    if maxh < minh %如果能持有的最大房产量 小于 下限
        ghouse2(:,ii:ii+nn*na*nh2-1) = [zeros(nh2-1,1);ghouse(i1_nh(ii))]*ones(1,nn*na*nh2); %要么不变，要么卖掉房子不要了
    elseif  ghouse(i1_nh(ii))>0  %如果现有房产大于能购买的最小值
        ghouse2(:,ii:ii+nn*na*nh2-1) = sort([0;exp(linspace(log(minh),log(maxh),nh2-2))';ghouse(i1_nh(ii))])*ones(1,nn*na*nh2);
    else
        ghouse2(:,ii:ii+nn*na*nh2-1) = [ghouse(i1_nh(ii));exp(linspace(log(minh),log(maxh),nh2-1))']*ones(1,nn*na*nh2);
    end

end

end


for ii=1:nn*na:numel(i4_nc)
ghouse2_1d(ii:ii+nn*na-1,1)  =  ghouse2(i2_nh(ii),ii);
end




for ii=1:numel(i4_nc)




if t >= tr-tb
                  
                     house_1(ii,1) = ghouse2(i2_nh(ii),ii)*gret_sh(i11_nn(ii),2); % 下一期的房产就是本期房产乘以收益率
                     house_1(ii,1) = max(min(house_1(ii,1),ghouse(nh)),ghouse(1));
                     
                     if ghouse2(i2_nh(ii),ii)==ghouse(i1_nh(ii)) %假如没调整房产
                         sav  = gcash(i3_ncash(ii))-gc(i4_nc(ii),ii)-ghouse(i1_nh(ii))*ppt;
                     else  %假如调整了房产
                         sav  = gcash(i3_ncash(ii))+ghouse(i1_nh(ii))*(1-adjcost-ppt)-gc(i4_nc(ii),ii)-ghouse2(i2_nh(ii),ii) ;
                     end

                     

                         if i5_na(ii)==1 %假如没有参与股票投资
                             cash_1(ii,1) = riskret(i5_na(ii),i11_nn(ii))*sav +ret_fac;
                             cash_1(ii,1) = max(min(cash_1(ii,1),gcash(ncash)),gcash(1));
                             %cash不能超过最大财富者，不能低于最小财富者
                         else %参与了股票投资
                             if sav  > ppcost + otcost
                             cash_1(ii,1) = riskret(i5_na(ii),i11_nn(ii))*(sav -ppcost-otcost)+ret_fac;
                             cash_1(ii,1) = max(min(cash_1(ii,1),gcash(ncash)),gcash(1));
                             else
                             cash_1(ii,1) = mincash;
                             end
                         end




else
                     house_1(ii,1) = ghouse2(i2_nh(ii),ii)*gret_sh(i11_nn(ii),2)/gyp; %下一期的房产就是本期房产乘以收益率
                     house_1(ii,1) = max(min(house_1(ii,1),ghouse(nh)),ghouse(1));
                     
                     if ghouse2(i2_nh(ii),ii)==ghouse(i1_nh(ii)) %假如没调整房产
                         sav  = gcash(i3_ncash(ii))-gc(i4_nc(ii),ii)-ghouse(i1_nh(ii))*ppt;
                     else  %假如调整了房产
                         sav  = gcash(i3_ncash(ii))+ghouse(i1_nh(ii))*(1-adjcost-ppt)-gc(i4_nc(ii),ii)-ghouse2(i2_nh(ii),ii);
                     end

                     if i5_na(ii)==1 %假如没有参与股票投资
                         cash_1(ii,1) = riskret(i5_na(ii),i11_nn(ii))*sav/gyp+1;
                         cash_1(ii,1) = max(min(cash_1(ii,1),gcash(ncash)),gcash(1));
                         %cash不能超过最大财富者，不能低于最小财富者
                     else %参与了股票投资
                         if sav  > ppcost + otcost
                         cash_1(ii,1) = riskret(i5_na(ii),i11_nn(ii))*(sav-ppcost-otcost)/gyp+1;
                         cash_1(ii,1) = max(min(cash_1(ii,1),gcash(ncash)),gcash(1));
                         else
                         cash_1(ii,1) = mincash;
                         end
                     end

    
end




end





%% 整个函数的END
end










