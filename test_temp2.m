load surv.mat 
gret_sh  = zeros(9,3);
[model]=griddedInterpolant({[1;3;9],[2;5;7]},[8,5,44;5,9,26;11,19,33],'spline');
[XOut, YOut, ZOut] = prepareSurfaceData([1;3;9], [2;5;7], [8,5,44;5,9,26;11,19,33]);
[model]=fit([XOut, YOut],ZOut,'cubicinterp'); 
param_cell = {5 5 0.95 -4 5 1 1 model 0.06 0.006 0 0 0.65 9 survprob gret_sh 1.01};% 一些需要传入目标函数的参数
y=my_auxV_cal([0.25 ,0.2,0],param_cell,5,10);