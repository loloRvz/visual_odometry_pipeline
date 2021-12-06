function I = warpImage(I_R, W)


I = zeros(size(I_R));

size_r = flip(size(I_R)); 

for x = 1:size(I_R,2)
    for y = 1:size(I_R,1)
        warped = (W * [x;y;1])';
        if all(warped < size_r & warped > [1 1])
            I(y, x) = I_R(int32(warped(2)), int32(warped(1)));
        end
    end
end