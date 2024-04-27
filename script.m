clearvars;
close all;
clc;

imgRGB = double(imread("image.png"));
imgHSV = double(rgb2hsv(uint8(imgRGB)));

[h, w, ~] = size(imgRGB);
[X, Y] = meshgrid(1:w, 1:h);
X = X / w;
Y = Y / h;
imgRGBxy = cat(3, imgRGB, X, Y);

[h, w, ~] = size(imgHSV);
[X, Y] = meshgrid(1:w, 1:h);
X = X / w;
Y = Y / h;
imgHSVxy = cat(3, imgHSV, X, Y);

K_values = [2, 4, 8, 16];
max_iter = 32;
for i = 1:length(K_values)
    K = K_values(i);
    
    [labels_rgb, ~] = kmeans_custom(imgRGB, K, max_iter);
    [labels_hsv, ~] = kmeans_custom(imgHSV, K, max_iter);
    [labels_rgbxy, ~] = kmeans_custom(imgRGBxy, K, max_iter);
    [labels_hsvxy, ~] = kmeans_custom(imgHSVxy, K, max_iter);

    figure;
    
    subplot(2, 2, 1);
    imshow(label2rgb(labels_rgb));
    title(['RGB - K = ', num2str(K)]);

    subplot(2, 2, 2);
    imshow(label2rgb(labels_hsv));
    title(['HSV - K = ', num2str(K)]);

    subplot(2, 2, 3);
    imshow(label2rgb(labels_rgbxy));
    title(['RGBxy - K = ', num2str(K)]);

    subplot(2, 2, 4);
    imshow(label2rgb(labels_hsvxy));
    title(['HSVxy - K = ', num2str(K)]);
end

function [labels, centroids] = kmeans_custom(X, K, max_iter)
    [H, W, ~] = size(X);
    X = reshape(X, H*W, []);
    
    centroids = X(randperm(H*W, K), :);
    
    for iter = 1:max_iter
        distances = zeros(H*W, K);
        for k = 1:K
            distances(:, k) = sum((X - centroids(k, :)).^2, 2);
        end
        [~, labels] = min(distances, [], 2);
        
        new_centroids = zeros(K, size(X, 2));
        for k = 1:K
            new_centroids(k, :) = mean(X(labels == k, :), 1);
        end
        
        if isequal(new_centroids, centroids)
            break;
        end
        
        centroids = new_centroids;
    end
    
    labels = reshape(labels, H, W);
end
