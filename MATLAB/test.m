function test()

dim =[ 
2    4    
8    16
16    32
32    64
64    128
128    256
256    512
512    1024
1024    2048
8192    16384
16384    32768
32768    65536
65536    131072
131072    262144];
m=size(dim);
    
    
for i=9:10%m(1)
    mn = dim(i,:);
    n = max(mn);
    m = min(mn);
 
    m2n = [m,n];
    x = strread(num2str(m2n),'%s');
    fid = fopen('/Users/Mohsen/Documents/nlpy_mohsen/lsq/matlab.txt','a+');
    
    [iter,primobj,gap,time,minimzer,history] = ...
        l1ls_test_problem(n,m,'random');
    %l1ls_test_problem(n,m,'partial_DCT_norm1');
%         
%     %
%     %
% 
l1ls = strcat(char(x(1)),char('--'),char(x(2)));

fprintf(fid,'%10s    &%-5d    &%-5.2e    &%-5.2e    &%-5.2f    &%-5d \n',l1ls,iter,primobj,gap, time, sum(history(5,:)));
%     
%    [~,IterN,realTol,objtrue,Time,Niter_lsmr] = ...
%        pdco_test_problem(n,m,'partial_DCT_norm1');
%        %pdco_test_problem(n,m,'random_norm1');
%       % 
%    PDCO = strcat(char(x(1)),char('--'),char(x(2)));
% fprintf(fid,'%s &%-2d& %-2.2e& %-2.2e& %-.2f& %-2d %s \n',PDCO,IterN,objtrue,realTol, Time, sum(Niter_lsmr),'\\');


fclose(fid);
end
end