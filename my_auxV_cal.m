function [ auxV ] = my_auxV_cal( myinput , param_cell , thecash ,thehouse )

%% 载入一些参数
t = param_cell{1};
rho  = param_cell{2};
delta  = param_cell{3};
psi_1  = param_cell{4};
psi_2  = param_cell{5};
theta  = param_cell{6};
gyp  = param_cell{7};
%V_next  = param_cell{8};
model  = param_cell{8};
adjcost  = param_cell{9};
ppt  = param_cell{10};
ppcost  = param_cell{11};
otcost  = param_cell{12};
income  = param_cell{13};
nn  = param_cell{14};
survprob  = param_cell{15};
gret_sh  = param_cell{16};
r  = param_cell{17};

%gcash  = param_cell{18};
%ghouse  = param_cell{19};

gcash1 = thecash;
ghouse1 = thehouse;

%% 载入待优化的参数
myc = myinput(1);
mya = myinput(2);
myh = myinput(3);


%% 计算消费对应的当期效用
u = (1.0-delta)*(myc^psi_1) ; %每种消费下的utility

%% 计算下期期初的cash和housing
% 计算housing下一期的nn种取值
housing_nn = zeros(nn,1);
for i = 1:nn
    housing_nn(i) = myh*gret_sh(i,2)/gyp ;
end

housing_nn(:) = max(min(housing_nn(:),19.9),0.25);


% 计算cash的值，分四种情况讨论
cash_nn = zeros(nn,1);

if myh~=ghouse1 && mya>0 %处置房产 且 参与股票
    sav  = gcash1+ghouse1*(1-adjcost-ppt)-myc-myh-ppcost-otcost;
    for i = 1:nn
        cash_nn(i) = (sav*(1-mya)*r + sav*mya*gret_sh(i,1))/gyp + income;
    end
end

if myh~=ghouse1 && mya==0 %处置房产 且 不参与股票
    sav  = gcash1+ghouse1*(1-adjcost-ppt)-myc-myh;
    cash_nn(:) = sav*r/gyp + income;
end

if myh==ghouse1 && mya>0 %不处置房产 且 参与股票
    sav  = gcash1+ghouse1*(-ppt)-myc-ppcost-otcost;
    for i = 1:nn
        cash_nn(i) = (sav*(1-mya)*r + sav*mya*gret_sh(i,1))/gyp + income;
    end
end

if myh==ghouse1 && mya==0 %不处置房产 且 不参与股票
    sav  = gcash1+ghouse1*(-ppt)-myc;
    cash_nn(:) = sav*r/gyp + income;
end

cash_nn(:) = max(min(cash_nn(:),19.9),0.25);

%% 插值到下一期的value function 计算未来效用
%int_V   =   interp2(ghouse(:),gcash(:),V_next,housing_nn(:),cash_nn(:),'spline'); 
int_V = model(housing_nn(:),cash_nn(:));
auxVV = (gret_sh(:,3)'*(int_V.^(1.0-rho)))'*survprob(t,1);
auxV = -(u+delta*(auxVV^(1.0/theta)))^psi_2;









end