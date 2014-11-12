max_seed = 100000;
num_per_seed = 10;
matlab_randoms = zeros(num_per_seed , max_seed+1);

for seed = 0: max_seed
    rng(seed);
    matlab_randoms(:,seed+1) = rand(num_per_seed,1);
end

save('matlab_randoms.mat','matlab_randoms')