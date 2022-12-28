% Q1()
Q2()

function [] = Q2() 
    img = imread("lena.jpg");
    img_lab = rgb2lab(img);
    L = mat2gray(img_lab(:,:,1)); %normalised L channel
    thresholds = [0.25, 0.5];
    minvar = 10000;
    minthresh = -1;
    for i = thresholds
        var_w = GetWithinClassVariance(L, i);
        if var_w<minvar
            minvar = var_w;
            minthresh = i;
        end
    end

    fprintf("\nMinimum within class variance: %f, found at threshold: %f\n", minvar, minthresh);
    L_thresh = imbinarize(L, minthresh);
    figure();
    subplot(2,2,1);
    imshow(img, []);
    title("RGB Image", "FontSize", 12);
    subplot(2,2,2);
    imshow(img_lab, []);
    title("LAB Image", "FontSize", 12);
    subplot(2,2,3);
    imshow(L, []);
    title("Normalized L Channel", "FontSize", 12);
    subplot(2,2,4);
    imshow(L_thresh, []);
    title("Segmented Image (L)", "FontSize", 12);

    imwrite(L_thresh, "q2_segmented_image.jpg", "jpg");
end

function [var_w] = GetWithinClassVariance(L, threshold)
    [pixel_count, bins] = imhist(L);
    pixel_prob = pixel_count./(size(L,1)*size(L,2));
    
    class1_prob = sum(L<=threshold, "all")/(size(L,1)*size(L,2));
    class2_prob = 1-class1_prob;
    class1_mean = 0;
    class2_mean = 0;
    class1_var = 0;
    class2_var = 0;
    
    %find class conditional mean
    for level = 1:size(bins)
        if (bins(level)<=threshold)
            class1_mean = class1_mean + (bins(level) * pixel_prob(level)/class1_prob);
        else
            class2_mean = class2_mean + (bins(level) * pixel_prob(level)/class2_prob);
        end
    end
    
    %find class conditional variance
    for level = 1:size(bins)
        if (bins(level)<=threshold)
            class1_var = class1_var + ((bins(level)-class1_mean)^2 * (pixel_prob(level)/class1_prob));
        else
            class2_var = class2_var + ((bins(level)-class2_mean)^2 * (pixel_prob(level)/class2_prob));    
        end
    end
    
    var_w = class1_var*class1_prob + class2_var*class2_prob;

    fprintf("\n\nFor threshold: %f\n", threshold);
    fprintf("Probability of class1: %f, for class2: %f\n", class1_prob, class2_prob);
    fprintf("Conditional mean for class1: %f, for class2: %f\n", class1_mean, class2_mean);
    fprintf("Conditional variance for class1: %f, for class2: %f\n", class1_var, class2_var);
    fprintf("Weighted within class variance: %f\n", var_w);

end

function [] = Q1()
    [img, map] = imread("palette-1c-8b.tiff"); %indexed image with colourmap
    img_rgb = ind2rgb(img, map);
    img_hsv = rgb2hsv(img_rgb);
    H = img_hsv(:,:,1);
    S = img_hsv(:,:,2);
    I = sum(img_rgb, 3)./3;
    
    img_hsi = cat(3, H, S, I);
    
    %perform canny edge detection on the I channel
    Sx = [-1 -2 -1; 0 0 0; 1 2 1];
    Sy = [-1 0 1; -2 0 2; -1 0 1];
    
    Mx = conv2(img_hsi(:,:,3), Sx, "same");
    My = conv2(img_hsi(:,:,3), Sy, "same");
    M = (Mx.^2 + My.^2).^0.5;
    G = atand(My./Mx);
    
    nm_supression = zeros(size(I));
    %if the angle is negative, add 360 to it and then check
    
    for i = 2:size(nm_supression, 1)-1
        for j = 2:size(nm_supression, 2)-1
            if G(i,j)<0
                G(i,j) = G(i,j)+360;
            end
    
            %angle 0
            if (G(i,j)>0 && G(i,j)<=22.5) || (G(i,j)>180 && G(i,j)<=202.5)
                if (M(i,j)>M(i, j+1) && M(i,j)>M(i, j-1))
                    nm_supression(i,j) = M(i,j);
                end
            end
    
            %angle 0
            if (G(i,j)>337.5 && G(i,j)<=360) || (G(i,j)>157.5 && G(i,j)<=180)
                if (M(i,j)>M(i, j+1) && M(i,j)>M(i, j-1))
                    nm_supression(i,j) = M(i,j);
                end
            end
            
            %angle 45
            if (G(i,j)>22.5 && G(i,j)<=67.5) || (G(i,j)>202.5 && G(i,j)<=247.5)
                if (M(i,j)>M(i+1, j+1) && M(i,j)>M(i-1, j-1))
                    nm_supression(i,j) = M(i,j);
                end
            end
    
            %angle 90
            if (G(i,j)>67.5 && G(i,j)<=112.5) || (G(i,j)>247.5 && G(i,j)<=292.5)
                if (M(i,j)>M(i+1, j) && M(i,j)>M(i-1, j))
                    nm_supression(i,j) = M(i,j);
                end
            end
    
            %angle 135
            if (G(i,j)>112.5 && G(i,j)<=157.5) || (G(i,j)>292.5 && G(i,j)<=337.5)
                if (M(i,j)>M(i+1, j-1) && M(i,j)>M(i-1, j+1))
                    nm_supression(i,j) = M(i,j);
                end
            end
        end
    end
    figure(); 
    subplot(2,4,1);
    imshow(img_rgb, []);
    title("RGB Image", "FontSize", 12)
    subplot(2,4,2);
    imshow(img_hsi, []);
    title("HSI Image", "FontSize", 12)
    subplot(2,4,3);
    imshow(Mx, []);
    title("X-Gradient Magnitude", "FontSize", 12)
    subplot(2,4,4);
    imshow(My, []);
    title("Y-Gradient Magnitude", "FontSize", 12)
    subplot(2,4,5);
    imshow(M, []);
    title("Gradient Magnitude", "FontSize", 12)
    subplot(2,4,6);
    imshow(G, []);
    title("Gradient Direction (in degrees)", "FontSize", 12)
    subplot(2,4,7);
    imshow(nm_supression, []);
    title("Non-Max Suppression", "FontSize", 12)

    imwrite(mat2gray(M), "q1_gradient_magnitude.tiff", "tiff");
    imwrite(mat2gray(Mx), "q1_gradient_magnitude_x.tiff", "tiff");
    imwrite(mat2gray(My), "q1_gradient_magnitude_y.tiff", "tiff");
    imwrite(mat2gray(G), "q1_gradient_direction.tiff", "tiff");
    imwrite(mat2gray(nm_supression), "q1_non_max_suppression.tiff", "tiff");
end



