function y = remap(x, min0, max0, min1, max1)
mean0 = .5*(max0+min0);
range0 = max0-min0;
mean1 = .5*(max1+min1);
range1 = max1-min1;
y = (x-mean0)*range1/range0 + mean1;