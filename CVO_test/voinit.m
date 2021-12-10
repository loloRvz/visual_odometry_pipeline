clc 
clear all;
close all;
%visual odometry initialization
%initializing parameters
addpath(".\plot")

% initialize the camera
IntrinsicMatrixkitti = [ 7.188560000000e+02 0 6.071928000000e+02
                    0 7.188560000000e+02 1.852157000000e+02
                    0 0 1];
IntrinsicMatrixpark = [331.37, 0,       320;
0,      369.568, 240;
0,      0,       1];
IntrinsicMatrix6=[1379.74 0 760.35;
    0 1382.08 503.41;
    0 0 1 ];
IntrinsicMatrix =IntrinsicMatrixpark';
cameraParams = cameraParameters('IntrinsicMatrix',IntrinsicMatrix);

% read first image
str = "..\data\parking\images\img_";
%str ="..\data\kitti\05\image_0\";
%str = "C:\Users\samso\OneDrive\Bureau\etude\VAMR\Exercise 6 - Two-view Geometry\data\";
I_R = rgb2gray(imread(sprintf('%s%05d.png',str,0)));
%selection of the index of the second frame 
idx = 8;

%extract features
%P_R = detectHarrisFeatures(I_R);
 P_R = detectSIFTFeatures(I_R);
% P_R = P_R.selectStrongest(200);
% num_features = 500;
% P_R = selectStrongest(P_R,num_features);
P_R = P_R.Location;

%initialization of the tracker 
trackerP = vision.PointTracker('MaxBidirectionalError',1);
initialize(trackerP,P_R,I_R);

% %tracking points
%track  intermediate frames
% %str = "C:\Users\samso\OneDrive\Bureau\etude\VAMR\Exercise 6 - Two-view Geometry\data\";
for i = 1:idx-1
    %read a neww frame
    I = rgb2gray(imread(sprintf('%s%05d.png',str,i)));

    %track point from old frame to new frame
    [~,~] = trackerP(I);
end
%track final frame
I = rgb2gray(imread(sprintf('%s%05d.png',str,idx)));
[P_F,keepP] = trackerP(I);
P_F = P_F(keepP,:);
P_R = P_R(keepP,:);

%essential matrix estimation 
[E,inliersIndex] = estimateEssentialMatrix(P_R,P_F,cameraParams);
P_F = P_F(inliersIndex,:);
P_R = P_R(inliersIndex,:);

%Camera pose extraction
[R_C_W,t_W_C] = relativeCameraPose(E,cameraParams,P_R,P_F);

%plot the points in the first and end frame that served for the pose
%extraction
figure (1)
imshow(I_R)
hold on
plot(P_R(:,1),P_R(:,2),"gx");
hold off
figure (2)
imshow(I)
hold on
plot(P_F(:,1),P_F(:,2),"gx");
hold off 

%triangulate 3d points
[P_i_1,X_i_1,C_i_1,f_i_1,T_i_1] = triangulate2dto3d(P_F,P_R,repmat([reshape(eye(3),1,[]),0,0,0],size(P_R,1),1),R_C_W, t_W_C,IntrinsicMatrix');
%test
% [R_C_W_1,t_W_C_1] = estimateWorldCameraPose(P_i_1,X_i_1,cameraParams);

%plot 3d points and camera poses
figure(3)
plot3(X_i_1(:,1),X_i_1(:,2),X_i_1(:,3),"mo");
plotCoordinateFrame(R_C_W', t_W_C', 2);
%plotCoordinateFrame(R_C_W_1', t_W_C_1', 2);
plotCoordinateFrame(eye(3), [0 0 0]', 2);

%plot points associated to the 3d points
figure (4)
imshow(I)
hold on
plot(P_i_1(:,1),P_i_1(:,2),"gx");
hold off


