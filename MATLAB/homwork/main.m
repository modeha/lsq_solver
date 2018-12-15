function  main()
load('c.mat')
load hw4image.txt;
dataset = hw4image;
centtroids = c;
c=[];
ctrn=[centtroids zeros(8,1)];
cent_0=[];

while (~isequal(cent_0,ctrn) )
 cent_0=ctrn;
for i=1:size(dataset,1)
c(i,1)=selectcentroid(dataset(i,:),ctrn(:,1:3));
end   
   
ctrn(1:8,4)=zeros(8,1);   
for i=1:size(c,1)   
ctrn(c(i,1),4)=ctrn(c(i,1),4)+1;
end
   
for i=1:size(ctrn,1)
if (ctrn(i,4) ~=0)
ctrn(i,1:3)=0;
end
end
 
for i=1:size(dataset,1)
ctrn(c(i,1),1:3)=ctrn(c(i,1),1:3)+dataset(i,1:3)/ctrn(c(i,1),4);
end

end


for i=1:size(dataset,1)
dataset(i,:)=ctrn(c(i,1),1:3);
end

display(ctrn(:,1:3));
display(ctrn(:,4));

mat = vec2mat(dataset,407);
imshow(uint8(mat));
end
function [ output_args ] = selectcentroid( sample,cent )

m=10000000; 
centroid=0;

for i=1:size(cent)
    dis= (sample(1)-cent(i,1))^2+(sample(2)-cent(i,2))^2+(sample(3)-cent(i,3))^2;
    if (m>dis)
        m=dis;
        centroid=i;
    end
end

output_args=centroid;

end
