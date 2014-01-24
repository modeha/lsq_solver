function dictInfo(x,x0,IterN,realTol,objtrue,time,name,type)
if strcmp(type ,'l1ls')
    if exist('l1ls.mat','file')
        load('l1ls','l1ls');
    else
        l1ls = java.util.Hashtable;
    end
    nnzminimizer = sparsity(x,1e-4);
    nnzoriginal  = sparsity(x0,1e-4);
    
    
    t=l1ls.get('Number of iteration');
    d = size(t);
    t(d+1) = IterN;
    l1ls.put('Number of iteration', t);
    
    
    t = l1ls.get('Name of problem');
    d = size (t);
    t(d+1) = name;
    l1ls.put('Name of problem',t);
    
    
    t = l1ls.get('Time');
    d = size (t);
    t(d+1) = time;
    l1ls.put('Time',t);
    
    t = l1ls.get('Objective');
    d = size (t);
    t(d+1) = objtrue;
    l1ls.put('Objective',t);
    
    t = l1ls.get('Number of non zero in minimizer');
    d = size (t);
    t(d+1) = nnzminimizer;
    l1ls.put('Number of non zero in minimizer',t);
    
    t = l1ls.get('Number of non zero in original');
    d = size (t);
    t(d+1) = nnzoriginal;
    l1ls.put('Number of non zero in original',t);
    
    t = l1ls.get('complementary');
    d = size (t);
    t(d+1) = realTol;
    l1ls.put('complementary',t);
 
    save('l1ls','l1ls');
else
    if exist('pdco.mat','file')
        load('pdco','pdco');
    else
        pdco = java.util.Hashtable;
    end
    
    nnzminimizer = sparsity(x,1e-4);
    nnzoriginal  = sparsity(x0,1e-4);
    
    t=pdco.get('Number of iteration');
    d = size(t);
    t(d+1) = IterN;
    pdco.put('Number of iteration', t);
    
    
    t = pdco.get('Name of problem');
    d = size (t);
    t(d+1) = 1;
    pdco.put('Name of problem',t);
    
    
    t = pdco.get('Time');
    d = size (t);
    t(d+1) = time;
    pdco.put('Time',t);
    
    t = pdco.get('Objective');
    d = size (t);
    t(d+1) = objtrue;
    pdco.put('Objective',t);
    
    t = pdco.get('Number of non zero in minimizer');
    d = size (t);
    t(d+1) = nnzminimizer;
    pdco.put('Number of non zero in minimizer',t);
    
    t = pdco.get('Number of non zero in original');
    d = size (t);
    t(d+1) = nnzoriginal;
    pdco.put('Number of non zero in original',t);
    
    t = pdco.get('complementary');
    d = size (t);
    t(d+1) = realTol;
    pdco.put('complementary',t);
    
%     t = pdco.get('primalinfeas');
%     d = size (t);
%     t(d+1) = Pinf;
%     pdco.put('primalinfeas',t);
%     
%     t = pdco.get('dualinfeas');
%     d = size (t);
%     t(d+1) = Dinf;
    pdco.put('gap',t);
    save('pdco','pdco');
end
end