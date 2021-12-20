clc;
clear all;
close all;

%% Setup
ds = 2; % 0: KITTI, 1: Malaga, 2: parking

kitti_path = './data/kitti';
malaga_path = './data/malaga-urban-dataset-extract-07';
parking_path = './data/parking';
addpath(genpath(pwd));

if ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
    assert(exist(kitti_path, 'dir') ~= 0);
    ground_truth = load([kitti_path '/poses/05.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    assert(exist(malaga_path, 'dir') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    % Path containing images, depths and all...
    assert(exist(parking_path, 'dir') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% Bootstrap
bootstrap_frames(1) = 0;
bootstrap_frames(2) = 3;

if ds == 0
    img0 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(1))]);
    img1 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(2))]);
elseif ds == 1
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(2))]));
else
    assert(false);
end

%Detect points img0
points0 = detectMinEigenFeatures(img0);
pointImage0 = insertMarker(img0,points0.Location,'+','Color','white');
% figure;
% imshow(pointImage0);
% title('Detected points img0');

%Init tracker
tracker = vision.PointTracker('MaxBidirectionalError',1);
initialize(tracker,points0.Location,img0);

%Track points to img1
[points1,validity] = tracker(img1);
pointImage1 = insertMarker(img1,points1(validity, :),'+');
% figure;
% imshow(pointImage1);
% title('Tracked points img1');

points0 = (points0(validity,:).Location');
points1 = (points1(validity,:)');

% figure;
% imshow(img1);
% hold on;
% plot(points1(1, :), points1(2, :), 'rx', 'Linewidth', 2);
% plot([points0(1,:); points1(1,:)], ...
%      [points0(2,:); points1(2,:)], ...
%      'g-', 'Linewidth', 3);
 
 
%Flip from matrix index to image coordinates
matched_points0 = points0;
matched_points1 = points1;

%Fundamental matrix
[F,inliersIndex] = estimateFundamentalMatrix(matched_points0',...
                                             matched_points1',...
                                             'Method','RANSAC'); 
%Essential matrix
E = K'*F*K;

%Only keep inliers
matched_points0 = matched_points0(:,inliersIndex);
matched_points1 = matched_points1(:,inliersIndex);

%Rotation, Translation
matched_points0 = [matched_points0;ones(1,length(matched_points0))];
matched_points1 = [matched_points1;ones(1,length(matched_points1))];
[R,u3] = decomposeEssentialMatrix(E);
[R_C_W,T_C_W] = disambiguateRelativePose(R,u3,matched_points0,matched_points1,K,K);

% Triangulate
M1 = K * eye(3,4);
M2 = K * [R_C_W, T_C_W];
P = linearTriangulation(matched_points0,matched_points1,M1,M2);

figure(3),
subplot(1,3,1)
plot3(P(1,:), P(2,:), P(3,:), 'o');
plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');

center_cam2_W = -R_C_W'*T_C_W;
plotCoordinateFrame(R_C_W',center_cam2_W, 0.8);
text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');

axis equal
rotate3d on;
grid

% Display matched points
subplot(1,3,2)
imshow(img0);
hold on
plot(matched_points0(1,:), matched_points0(2,:), 'ys');
title('Image 0')

subplot(1,3,3)
imshow(img1);
hold on
plot(matched_points1(1,:), matched_points1(2,:), 'ys');
title('Image 1')


%% Continuous operation
range = (bootstrap_frames(2)+1):last_frame;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([kitti_path '/05/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        image = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    % Makes sure that plots refresh.    
    pause(0.01);
    
    prev_img = image;
end
