%% 说明
% Quantitative decomposition
% 如果给low的人 high的参数
% y能被多解释多少
clear
xlow =[0.36106      0.70974      0.36162      0.80249      0.66785      0.38534      0.38451      0.34775];
xhigh=[0.10646      0.24896      0.12587      0.75735      0.61842      0.65657      0.43493      0.39775];
[~,~,bt1]=my_estimation_prepostdid1_high(xhigh);
[~,~,bt2]=my_estimation_prepostdid1_low(xhigh);
[~,~,bt3]=my_estimation_prepostdid1_low(xlow);
result=zeros(length(bt1(:,1)),length(xhigh));
for i=1:length(xhigh)
    xtemp=xlow;
    xtemp(i)=xhigh(i);
    [~,~,bt2]=my_estimation_prepostdid1_low(xtemp);
    result(:,i)=bt2(:,2);
end
%beta=[beta1;beta2;beta3];
[bt1(:,2),bt2(:,2)];
[bt1(:,2),result(:,:)];
save decomp.mat bt1 bt2 bt3 result
load decomp.mat
ratio=zeros(length(bt1(:,1)),length(xhigh));
for i=1:length(xlow)
    ratio(:,i)=1-(bt1(:,2)-result(:,i))./(bt1(:,2)-bt3(:,2));
end
decomp1=mean(ratio,1)';




%% 把财富也赋过去
load mySample_pre10.mat %既包含上海 也包含非上海
mySample_low =mySample(mySample(:,9)==0,:);
mySample_high=mySample(mySample(:,9)==1,:);

ratio =  mean(mySample_high(:,4))/mean(mySample_low(:,4));
ratio2 = median(mySample_high(:,4))/median(mySample_low(:,4));

mySample_low(:,4)=mySample_low(:,4)*ratio;
mySample_low(:,5)=mySample_low(:,5)*ratio;
mySample = [mySample_low;mySample_high];
save mySample_pre10_decomp_wealth.mat mySample


%% 
%[y1,z1,bt1]=my_estimation_prepostdid1_high(xhigh);
[~,~,bt2_wealth]=decomposition_my_estimation_prepostdid1_low(xhigh);
%[y3,z3,bt3]=my_estimation_prepostdid1_low(xlow);
result=zeros(length(bt1(:,1)),length(xhigh));
for i=1:length(xhigh)
    xtemp=xlow;
    xtemp(i)=xhigh(i);
    [~,~,bt2_wealth]=my_estimation_prepostdid1_low(xtemp);
    result(:,i)=bt2_wealth(:,2);
end
beta=[beta1;beta2;beta3];
[bt1(:,2),bt2_wealth(:,2)];
[bt1(:,2),result(:,:)];
save decomp1.mat bt1 bt2_wealth bt3 result
load decomp1.mat
ratio=zeros(length(bt1(:,1)),length(xhigh));
for i=1:length(xlow)
    ratio(:,i)=1-(bt1(:,2)-result(:,i))./(bt1(:,2)-bt3(:,2));
end
decomp2=mean(ratio,1)';









%% Reduced
clear
xlow_reduce =[0.26122      0.88806      0.25246      0.89596      0.35241];
xhigh_reduce=[0.25315      0.215708     0.30468      0.71         0.72634];
[y1,z1,bt1]=my_estimation_prepostdid1_high(xhigh_reduce);
[y2,z2,bt2]=my_estimation_prepostdid1_low(xhigh_reduce);
[y3,z3,bt3]=my_estimation_prepostdid1_low(xlow_reduce);
result=zeros(length(bt1(:,1)),length(xhigh_reduce));
for i=1:length(xhigh_reduce)
    xtemp=xhigh_reduce;
    xtemp(i)=xhigh_reduce(i);
    [y2,z2,bt2]=my_estimation_prepostdid1_low(xtemp);
    result(:,i)=bt2(:,2);
end
beta=[beta1;beta2;beta3];
[bt1(:,2),bt2(:,2)];
[bt1(:,2),result(:,:)];
save decomp2.mat bt1 bt2 bt3 result
load decomp2.mat
ratio=zeros(length(bt1(:,1)),length(xhigh_reduce));
for i=1:length(xhigh_reduce)
    ratio(:,i)=1-(bt1(:,2)-result(:,i))./(bt1(:,2)-bt3(:,2));
end
decomp3=mean(ratio,1)';










