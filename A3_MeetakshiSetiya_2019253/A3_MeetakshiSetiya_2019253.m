%Meetakshi Setiya, 2019253
%Assignment 3

% q1()
% q2()
 for i = [20, 30, 50]
     q3(i)
end

%Q3: Frequency domain filtering
function [] = q3(k)
img_original = imresize(imread("cameraman.jpg"), 2);
img_noisy = img_original(:, :);

%add noise
for i = [0, 100, 200, 300, 400, 500]
    for j = 0:511
        img_noisy(i+1, j+1) = img_noisy(i+1, j+1)+k;
    end
end

%a modified ideal lowpass filter with almost zero pixel values in the
%center
filter = ones(size(img_noisy));
for i = 0:511
     if abs(i-256)<15 %ideal lowpass
         continue
     end
    filter(i+1, 256) = 0.1;
    filter(i+1, 255) = 0.1;
    filter(i+1, 257) = 0.1;
end

fft_original = fft2(double(img_original));
shiftedfft_original = fftshift(fft_original);

fft_noisy = fft2(double(img_noisy));
shiftedfft_noisy = fftshift(fft_noisy);

shiftedfft_denoised = shiftedfft_noisy.*filter; %hadamard product
fft_denoised = ifftshift(shiftedfft_denoised);
img_denoised = real(ifft2(fft_denoised));

figure()
sgtitle(sprintf("Q3, K = %d", k), 'FontSize', 20)
subplot(1,3,1);
imshow(img_original);
title('Original Image','FontSize', 12);
subplot(1,3,2);
imshow(img_noisy);
title('Noisy Image','FontSize', 12);
subplot(1,3,3);
imshow(img_denoised, []);
title('Denoised Image','FontSize', 12);

figure()
sgtitle(sprintf("Q3, K = %d", k), 'FontSize', 20)
subplot(2,2,1);
imshow(log(abs(shiftedfft_original)), []);
title('Magnitude Spectrum of Original Image','FontSize', 12);
subplot(2,2,2);
imshow(log(abs(shiftedfft_noisy)), []);
title('Magnitude Spectrum of Noisy Image','FontSize', 12);
subplot(2,2,3);
imshow(filter);
title('Filter','FontSize', 12);
subplot(2,2,4);
imshow(log(abs(shiftedfft_denoised)), []);
title('Magnitude Spectrum of Denoised Image','FontSize', 12);

i1 = log(abs(shiftedfft_noisy));
i1 = i1 - min(i1(:));
i1 = i1/max(i1(:));

i2 = log(abs(shiftedfft_denoised));
i2 = i2 - min(i2(:));
i2 = i2/max(i2(:));

i3 = log(abs(shiftedfft_original));
i3 = i3 - min(i3(:));
i3 = i3/max(i3(:)); 

dn = img_denoised - min(img_denoised(:));
dn = dn/max(dn(:));

imwrite(img_noisy, sprintf("q3_noisy_image_k%d.jpeg", k));
imwrite(dn, sprintf("q3_denoised_image_k%d.jpeg", k));
imwrite(i1, sprintf("q3_mag_spectrum_noisy_image_k%d.jpeg", k));
imwrite(i2, sprintf("q3_mag_spectrum_denoised_image_k%d.jpeg", k));
imwrite(i3, "q3_mag_spectrum_original_image.jpeg");
imwrite(filter, "q3_filter.jpeg");
end

%Q2: Masking in frequency domain
function [] = q2()
    img = im2double(imread("x5.bmp"));
    box_filter = (1/25) * ones(5);
    f_zeropadded = padarray(img, size(box_filter)-1, 'post');
    w_zeropadded = padarray(box_filter, size(img)-1, 'post');
    F = fft2(f_zeropadded);
    W = fft2(w_zeropadded);
    mul_FW = F.*W;
    mask = F-mul_FW;
    Unsharp = F+mask;
    Highboost = F+4*mask;
    unsharp = real(ifft2(Unsharp));
    highboost = real(ifft2(Highboost));
    final_unsharp = imcrop(unsharp, [0, 0, 512, 512]);
    final_highboost = imcrop(highboost, [0, 0, 512, 512]);

    figure()
    sgtitle("Q2", 'FontSize', 20)
    subplot(1,3,1);
    imshow(img);
    title('Original Image','FontSize', 12);
    subplot(1,3,2);
    imshow(final_unsharp);
    title("Unsharp Masked Image",'FontSize', 12); %matlab already caps negative values to 0 and >255 to 255
    subplot(1,3,3);
    imshow(final_highboost);
    title("Highboost Filtered Image",'FontSize', 12);
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

    imwrite(final_unsharp, "q2_unsharp_masked_image.bmp", "bmp");
    imwrite(final_highboost, "q2_highboost_filtered_image.bmp", "bmp");
end

%Q1: Unsharp Masking
function [] = q1()
    img = im2double(imread("x5.bmp"));
    box_filter = (1/25) * ones(5);
    convolved_img = conv2(img, box_filter, 'same');
    mask = img-convolved_img;
    masked_img = img+mask;
    highboost_img = img+4*mask;

    figure()
    sgtitle("Q1", 'FontSize', 20)
    subplot(2,2,1);
    imshow(img);
    title('Original Image','FontSize', 12);
    subplot(2,2,2);
    imshow((convolved_img - min(convolved_img(:))/max(highboost_img(:))));
    title('Blurred Image','FontSize', 12);
    subplot(2,2,3);
    imshow(masked_img);
    title("Unsharp Masked Image",'FontSize', 12);
    subplot(2,2,4);
    imshow(highboost_img);
    title("Highboost Filtered Image",'FontSize', 12);
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

    imwrite((convolved_img - min(convolved_img(:))/max(highboost_img(:))), "q1_blurred_image.bmp", "bmp");
    imwrite(masked_img, "q1_unsharp_masked_image.bmp", "bmp");
    imwrite(highboost_img, "q1_highboost_filtered_image.bmp", "bmp");
end
