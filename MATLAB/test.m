function test(type,solver)
epsilon  = 0.01;
dim =[ 
2 8
4 16
8 32
16 64
32 128
64 256
128 512
256 1024
512 2048
1024 4096
2048 8192
4096 16384
8192 32768
16384 65536
32768 131072
65536 262144
131072 524288
262144 1048576];
m=size(dim);
   
    
for i=16:18
    %mn = dim(i,:);
    mn = dim(i,:);
    n = max(mn);
    m = min(mn);
    m2n = [m,n];
    x = strread(num2str(m2n),'%s');
    fid = fopen('/Users/Mohsen/Documents/nlpy_mohsen/lsq/matlab.txt','a+');
   
    if strcmp(solver,'l1ls')
        [iter,primobj,gap,time,minimzer,history,status] = ...
            l1ls_test_problem(n,m,type,epsilon);
    l1ls = strcat(char(x(1)),char('--'),char(x(2)));
    fprintf(fid,'%-15s  &%-5d  &%-5.2e  &%-5.2e  &%-5.2f  &%-5d %s\n'...
        ,l1ls,iter,primobj,gap, time, sum(history(5,:)),'\\');
    status,l1ls
    %norm_x0_x(minimzer,epsilon)
    else
        if strcmp(type,'DCT')
            gamma   = 19;
        else
            gamma   = .0019;
        end
        [minimzer,IterN,realTol,objtrue,Time,Niter_lsmr] = ...
            pdco_test_problem(n,m,type,epsilon,gamma);
   PDCO = strcat(char(x(1)),char('--'),char(x(2)))
   fprintf(fid,'%-15s &%-5d& %-5.2e& %-5.2e& %-5.2f& %-5d %s\n',...
       PDCO,IterN,objtrue,realTol, Time, sum(Niter_lsmr),'\\');
   %norm_x0_x(minimzer,epsilon);
    end
fclose(fid);
end
end

function n=norm_x0_x(minimzer,epsilon)
t = size(minimzer);
 x0 = parameterD(t(1),epsilon); 
 n =norm(x0-minimzer);
end