%% DID 第二种方法 all
all = [0.2190    0.0434    0.7759    0.9900    0.9331];


theta   = reshape(all,1,5).*[10000,200000,8,0.29,0.4]+[0,0,2,0.7,0.3];
nmoments = 27;
load Sample_did.mat
l = length(mySample);

dtv = [30,19000,0.5,0.01,0.01];
ntheta = 5;
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ~ , g0(:,1) , xdid ] = my_estimation_prepostdid1(theta);

for i = 1:ntheta
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1(theta1);
    [ ~ , g2(:,i)] = my_estimation_prepostdid1(theta2);

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end

cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_did1 = sqrt(diag(cov))';
overid_did1 = l*9/(1+9)*g0(:,1)'*W*g0(:,1)
theta
se_did1

%% did 第二种方法 - low
low = [0.1275    0.0968    0.9853    0.9484    0.6653];

theta   = reshape(low,1,5).*[10000,200000,8,0.29,0.4]+[0,0,2,0.7,0.3];
nmoments = 27;
load Sample_did_low.mat
l = length(mySample);

dtv = [30,2000,0.5,0.01,0.01];
ntheta = 5;
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ~ , g0(:,1) , xdidl ] = my_estimation_prepostdid1_low(theta);

parfor i = 1:ntheta
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1_low(theta1);
    [ ~ , g2(:,i)] = my_estimation_prepostdid1_low(theta2);

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end

cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_did1low = sqrt(diag(cov))';
overid_did1low = l*9/(1+9)*g0(:,1)'*W*g0(:,1)
theta
se_did1low



%% DID 第二种方法 high
high = [0.1508    0.0115    0.6262    0.9589    0.8936 ];


theta   = reshape(high,1,5).*[10000,200000,8,0.29,0.4]+[0,0,2,0.7,0.3];
nmoments = 27;
load Sample_did_high.mat
l = length(mySample);

dtv = [30,2200,0.5,0.01,0.01];
ntheta = 5;
g1 = zeros(nmoments,ntheta);
g2 = zeros(nmoments,ntheta);
g0 = zeros(nmoments,1);
G  = zeros(nmoments,ntheta);

[ ~ , g0(:,1) , xdidh ] = my_estimation_prepostdid1_high(theta);

for i = 1:ntheta
    dt = dtv(1,i);
    theta1  = theta;
    theta1(1,i)  = theta1(1,i) + dt;

    theta2  = theta;
    theta2(1,i)  = theta2(1,i) - dt;

    [ ~ , g1(:,i)] = my_estimation_prepostdid1_high(theta1);
    [ ~ , g2(:,i)] = my_estimation_prepostdid1_high(theta2);

    G(:,i) = (g1(:,i) - g2(:,i))./(2*dt);

end

cov = 1/(l) * (1 + 1/9) * pinv(G'*W*G) ;
se_did1high = sqrt(diag(cov))';
overid_did1high = l*9/(1+9)*g0(:,1)'*W*g0(:,1)
theta
se_did1high




