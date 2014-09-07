function test(number,Solver)
k=0;
for i=1:number
    
  %PDCOTest(1024,1280,1)
  realTol = PDCOTest(120,190,Solver);
  if realTol > 1e-4
      k = k+1;
  end
end
k
end