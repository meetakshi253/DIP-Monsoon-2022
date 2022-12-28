rng(10);
img = imread('x5.bmp');
noisy_img = (sqrt(100)*randn(size(img)))+0 + double(img);
[a,bar] = CalculatePSNR(img, mat2gray(noisy_img));
disp(a);
% q1(img, noisy_img)
q2a(img, noisy_img)
% q2b(img, noisy_img)

function[] = q2b(img, noisy_img)
    gaussian_x = fspecial('gaussian', [1,13], 2);
    gaussian_y = fspecial('gaussian', [13,1], 2);
    laplacian_x = [1 -2 1];
    laplacian_y = laplacian_x';
    
    blurred_x = conv2(mat2gray(noisy_img), gaussian_x, 'same');
    blurred_xy = conv2(blurred_x, gaussian_y, 'same');
    response_x = conv2(blurred_xy, laplacian_x, 'same');
    response_y = conv2(blurred_xy, laplacian_y, 'same');
    response = response_x+response_y;
    
    response_norm = mat2gray(response); %no need to do x-min/max
    
    response_2a = imread("q2a_convolved_img.bmp");
    difference = abs(mat2gray(response_2a)-response_norm);
    sod = sum(difference(:));
    fprintf("Sum of absolute of difference: %f\n", sod);
    fprintf("Mean sum of absolute of difference: %f\n", sod/(512*512));
    
    response_255 = uint8(255*response_norm);
    difference = abs(response_2a-response_255);
    sod = sum(difference(:));
    fprintf("Sum of absolute of difference (255): %f\n", sod);
    fprintf("Mean sum of absolute of difference: %f\n", sod/(512*512));
    
    zero_crossed = ZeroCrossing(response, 0.04);
    
    figure()
    subplot(3,3,1);
    imshow(img, []);
    title("Clean image", 'FontSize', 12);
    subplot(3,3,2);
    imshow(noisy_img, []);
    title("Noisy Image", 'FontSize', 12);
    subplot(3,3,3);
    imshow(gaussian_x, []);
    title("Horizontal 1D gaussian filter", 'FontSize', 12);
    subplot(3,3,4);
    imshow(gaussian_y, []);
    title("Vertical 1D gaussian filter", 'FontSize', 12);
    subplot(3,3,5);
    imshow(blurred_xy, []);
    title("Response after gaussian filtering", 'FontSize', 12);
    subplot(3,3,6);
    imshow(response_x, []);
    title("Response after hz second order derivative filtering", 'FontSize', 12);
    subplot(3,3,7);
    imshow(response_y, []);
    title("Response after vert second order derivative filtering", 'FontSize', 12);
    subplot(3,3,8);
    imshow(response, []);
    title("Combined response after second order derivative filtering", 'FontSize', 12);
    subplot(3,3,9);
    imshow(zero_crossed, []);
    title("Edges after zero crossing", 'FontSize', 12);
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    
    imwrite(mat2gray(blurred_xy), "q2b_gaussian.bmp", "bmp");
    imwrite(mat2gray(response_x), "q2b_hz_2derv.bmp", "bmp");
    imwrite(mat2gray(response_y), "q2b_vert_2derv.bmp", "bmp");
    imwrite(mat2gray(response), "q2b_2derv.bmp", "bmp");
    imwrite(mat2gray(zero_crossed), "q2b_zero_crossing.bmp", "bmp");
    imwrite(mat2gray(laplacian_x), "q2b_hz_2derv_filter.bmp", "bmp");
    imwrite(mat2gray(laplacian_y), "q2b_vert_2derv_filter.bmp", "bmp");
end


function [] = q2a(img, noisy_img)
    log_filter = fspecial("log", 13, 2); %filter size > 6*sigma
    convolved_img = conv2(noisy_img, log_filter, 'same');
    response = ZeroCrossing(convolved_img, 0.04);
    
    figure()
    subplot(2,3,1);
    imshow(img, []);
    title("Clean image", "FontSize", 12);
    subplot(2,3,2);
    imshow(noisy_img, []);
    title("Noisy image", "FontSize", 12);
    subplot(2,3,3);
    imshow(log_filter, []);
    title("LoG filter", "FontSize", 12);
    subplot(2,3,4);
    imshow(convolved_img, []);
    title("Response after convolution", "FontSize", 12);
    subplot(2,3,5);
    imshow(response, []);
    title("Edges after zero-crossing", "FontSize", 12);
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    
    imwrite(mat2gray(log_filter), "q2a_log_filter.bmp", "bmp");
    imwrite(mat2gray(convolved_img), "q2a_convolved_img.bmp", "bmp");
    imwrite(mat2gray(response), "q2a_zero_crossing.bmp", "bmp");
end

function[] = q1(img, noisy_img)
    k = [10e-4, 10e-3, 10e-2, 10e-1, 10e1, 10e2, 10e3, 10e4];
    gamma = [0.01, 0.1, 1, 10e1, 10e2, 10e3, 10e4];
    
    %expression for W = H*/(|H|^2 + k)(1+g|L|^2) where g = gamma
    %taking L as a 3x3 laplacian filter and H as a 3x3 Gaussian filter
    
    laplacian = [0 1 0; 1 -4 1; 0 1 0];
    gaussian = 1/16*[1 2 1; 2 4 2; 1 2 1];
    noisy_img_padded = padarray(noisy_img, size(laplacian)-1, 'post');
    laplacian_padded = padarray(laplacian, size(noisy_img)-1, 'post');
    gaussian_padded = padarray(gaussian, size(noisy_img)-1, 'post');
    
    %take FFT of the matrices
    H = fft2(gaussian_padded);
    L = fft2(laplacian_padded);
    G = fft2(noisy_img_padded);
    H_conj = conj(H);
    
    %use gridsearch to find the optimal gamma and k
    maxpsnr = -1;
    opt_k = -1;
    opt_gamma = -1;
    opt_img = img;
    for i = k
        for jj = gamma
            W = H_conj./((abs(H)^2 + i).*(1+jj*abs(L)^2));
            F_cap = W.*G;
            denoised_img = real(ifft2(F_cap));
            denoised_img = denoised_img - min(denoised_img(:));
            denoised_img = denoised_img/max(denoised_img(:));
            denoised_img = imcrop(denoised_img, [0, 0, 512, 512]);
            [peaksnr, f_cap_scaled] = CalculatePSNR(img, denoised_img);
            if (peaksnr>maxpsnr)
                maxpsnr = peaksnr;
                opt_k = i;
                opt_gamma = jj;
                opt_img = f_cap_scaled;
            end
        end
    end
    fprintf("Best PSNR for custom filter: %f. Found at k=%d, gamma=%d.\n", maxpsnr, opt_k, opt_gamma);

    maxpsnrw = -1;
    opt_kw = -1;
    opt_imgw = img;
    %do the same for wiener filter. gamma = 0
    for i = k
        W = H_conj./(abs(H)^2 + i);
        F_cap = W.*G;
        denoised_img = real(ifft2(F_cap));
        denoised_img = denoised_img - min(denoised_img(:));
        denoised_img = denoised_img/max(denoised_img(:));
        denoised_img = imcrop(denoised_img, [0, 0, 512, 512]);
        [peaksnr, f_cap_scaled] = CalculatePSNR(img, denoised_img);
        if (peaksnr>maxpsnrw)
           maxpsnrw = peaksnr;
           opt_kw = i;
           opt_imgw = f_cap_scaled;
        end
    end

    fprintf("Best PSNR for wiener filter: %f. Found at k=%d\n", maxpsnrw, opt_kw);

    figure()
    subplot(2,2,1);
    imshow(img, []);
    title("True image", 'FontSize', 12);
    subplot(2,2,2);
    imshow(noisy_img, []);
    title("Noisy Image with AWGN", 'FontSize', 12);
    subplot(2,2,3);
    imshow(opt_img, []);
    title("Denoised Image with custom filter", 'FontSize', 12);
    subplot(2,2,4);
    imshow(opt_imgw, []);
    title("Denoised Image with wiener filter", 'FontSize', 12);
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    imwrite((noisy_img - min(noisy_img(:)))/max(noisy_img(:)), "q1_noisy.bmp", "bmp");
    imwrite(opt_img, "q1_denoised_custom.bmp", "bmp");
    imwrite(opt_imgw, "q1_denoised_wiener.bmp", "bmp");
end


function [response] = ZeroCrossing(img, thresh)
    padded_img = padarray(img, [1,1], 0, 'both');
    response = zeros(size(img));
    threshold = thresh*max(img(:));
    for ii = 2:size(padded_img, 1)-1
        for jj = 2:size(padded_img, 2)-1
            neighbours = [padded_img(ii,jj-1), padded_img(ii,jj+1), padded_img(ii-1,jj), padded_img(ii+1,jj), ...
                padded_img(ii+1,jj+1), padded_img(ii-1,jj-1), padded_img(ii+1,jj-1), padded_img(ii-1,jj+1)];
            if(neighbours(1)*neighbours(2)<0 && abs(neighbours(1)-neighbours(2))>threshold)
                response(ii-1,jj-1)=1;
            elseif(neighbours(3)*neighbours(4)<0 && abs(neighbours(3)-neighbours(4))>threshold)
                response(ii-1,jj-1)=1;
            elseif(neighbours(5)*neighbours(6)<0 && abs(neighbours(5)-neighbours(6))>threshold)
                response(ii-1,jj-1)=1;
            elseif(neighbours(7)*neighbours(8)<0 && abs(neighbours(7)-neighbours(8))>threshold)
                response(ii-1,jj-1)=1;
            end
        end
    end
end


function [peaksnr, f_cap_scaled] = CalculatePSNR(f, f_cap)
    f_scaled = f-min(f(:));
    f_scaled = f/max(f_scaled(:));
    f_scaled = uint8(round(f_scaled*255));
    f_cap_scaled = uint8(round(f_cap*255));  
    mse = mean((f_scaled-f_cap_scaled).^2, 'all');
    peaksnr = 10*log10(255^2/mse);
end
