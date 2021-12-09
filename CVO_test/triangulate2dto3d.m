function [newP,newX,C_i,f_i,T_i] = triangulate2dto3d(C_i,f_i,T_i,R_C_W, t_W_C,K)
    newP = zeros(size(C_i));
    newX = zeros(size(C_i,1),3);
    keep = true(size(C_i,1),1);
    TCW = -R_C_W*t_W_C';
    M2 = K*[R_C_W TCW];
    parfor i = 1:size(C_i,1)
       % if (norm(C_i(i,:)-f_i(i,:)))>25
        RWC_i = reshape(T_i(i,1:9),3,3);
        TWC_i = T_i(i,10:12)';
        TCW_i = -RWC_i'*TWC_i;
        M1 = K*[RWC_i' TCW_i];
        p1x = cross2Matrix([f_i(i,:)';1]);
        p2x = cross2Matrix([C_i(i,:)';1]);
        A = [p1x*M1;p2x*M2];
        [~,~,V] = svd(A);
        P = V(:,end);
        P = P (1:3)/P (end);
        P1 = R_C_W*(P-t_W_C');
        P2 = RWC_i'*(P-TWC_i);
        if (P1(3)<=0.5|| P2(3)<=0.5)
            continue
        end
%         csalpha = (P-t_W_C')'*(P-TWC_i)/norm(P-t_W_C')/norm(P-TWC_i);
        if abs(atan2(norm(cross(P-t_W_C',P-TWC_i)), dot(P-t_W_C',P-TWC_i)))>1.5*pi/180
        keep(i) = false;
        newP(i,:) = C_i(i,:);
        newX(i,:) = P';
        end
    end
    newP = newP(not(keep),:);
    newX = newX(not(keep),:);
    C_i = C_i(keep,:);
    f_i = f_i(keep,:);
    T_i = T_i(keep,:);
end