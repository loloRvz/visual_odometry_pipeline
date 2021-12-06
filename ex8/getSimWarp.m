function W = getSimWarp(dx, dy, alpha_deg, lambda)

alpha_rad = alpha_deg / 180 * pi;

c = cos(alpha_rad);
s = sin(alpha_rad);
M = [c,-s;
     s,c];
 
W = lambda * [M, [dx;dy]];

end
