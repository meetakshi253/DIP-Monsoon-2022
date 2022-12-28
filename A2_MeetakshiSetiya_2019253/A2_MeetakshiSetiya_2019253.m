Img = imread('x5.bmp');
I = imcrop(Img, [0, 0, 200, 200]);
T_t = [1 0 0; 0 1 0; 20 20 1];
T_r = [cosd(30) -sind(30) 0; sind(30) cosd(30) 0; 0 0 1]; %10 degrees clockwise
T = T_t*T_r;
q1_o = q1(I, T);
q2_o = q2(I, q1_o, T);
q3(Img);

function [out_v] = bilinear_interpolation(img, x, y)
     [rows, cols] = size(img);
     x1 = floor(x);
     y1 = floor(y);
     x2 = x1+1;
     y2 = y1+1;
     V = [0 0 0 0]; %zero padding by default. if valid pixel value exists, replace the 0.
     count = 0;
     for i = x1:x2
         for j = y1:y2
             count = count+1;
             if i<0 || j<0 || i>rows-1 || j>cols-1
                 continue;
             else
                 V(count) = img(i+1, j+1);
             end
         end
     end
     X = [x1, y1, x1*y1, 1; x1, y2, x1*y2, 1; x2, y1, x2*y1, 1; x2, y2, x2*y2, 1];
     A = X\V';
     out_v = A(1)*x + A(2)*y + A(3)*x*y + A(4);
end

function [output] = q1(I, T)
    [rows, cols] = size(I);
    output = zeros(floor(rows*1.5), floor(cols*1.5));
    [out_rows, out_cols] = size(output);

    for i = 0:out_rows-1
        for j = 0:out_cols-1
            m = [i, j, 1]/T;
            x = m(1); y = m(2);
            output(i+1, j+1) = bilinear_interpolation(I, x, y);
        end
    end
    output = output - min(output(:));
    output = output / max(output(:));
    figure, imshow(output);
    title('Image after transformation','FontSize', 12);
    imwrite(output, 'q1 transformation.bmp');
end

function [Ureg] = q2(U, X, T) %U is original image wrt which the image should be registered, X is unregistered image
    Ureg = zeros(size(U));
    [out_rows, out_cols] = size(Ureg);

    for i = 0:out_rows-1
        for j = 0:out_cols-1
            m = [i, j, 1]*T; 
            x = m(1); y = m(2);
            Ureg(i+1, j+1) = bilinear_interpolation(X, x, y);
        end
    end
    figure, imshow(Ureg);
    title('Registered Image','FontSize', 12);
    imwrite(Ureg, 'q2 registration.bmp');
end

function [H, G] = q3(I)
    I_log_transformed = log_transform(I);
    values = make_normalised_histogram(I, 'q3 Histogram OG Image');
    values_log = make_normalised_histogram(I_log_transformed, 'q3 Histogram Log-Transf Image');

    %q3, part c
    H = (255*cumsum(values));
    G = (255*cumsum(values_log));
    plot(H, '-', 'LineWidth', 2);
    hold on;
    plot(G, '-', 'LineWidth', 2);
    grid on;
    hold off;
    legend('Original Image','Log Transformed Image');
    xlabel('CDF', 'FontSize', 10);
    ylabel('Pixel Count', 'FontSize', 10);
    title('CDFs', 'FontSize', 12);
    savefig('q3 CDF.fig');

    %q3, part d
    argmin_s = zeros(1, 256);
    for r = 1:256 %for some input pixel r
        arr = double(abs(H(r)-G));
        m = min(arr); 
        pos = find(arr==m, 1); %find the first index with min value
        argmin_s(r) = pos;
    end

    %q3, part e
    I_matched = zeros(size(I));
    for i=1:size(I,1)
        for j=1:size(I,2)
            I_matched(i,j) = argmin_s(I(i,j));
        end
    end
    figure, imshow(I_matched/255);
    title('Matched Image', 'FontSize', 12);
    imwrite(I_matched/255, 'q3 Matched.bmp');
end


%q3, part b
function [values] = make_normalised_histogram(I, Title)
    values = zeros(1, 256);
    for i = 0:255
        values(i+1) = sum(I(:) == i);
    end
    values = values/(size(I,1)*size(I,2));
    gray_levels = 0:255;
    figure, bar(gray_levels, values, 'BarWidth', 0.5);
    xlabel('Gray Level', 'FontSize', 10);
    ylabel('Pixel Count', 'FontSize', 10);
    title(Title, 'FontSize', 12);
    grid on;
    savefig(sprintf('%s.fig', Title));
end

%q3, part a
function [I_log_transformed] = log_transform(I)
   [rows, cols] = size(I);
   I_log_transformed = zeros(rows, cols);
   for i = 1:rows
       for j = 1:cols
           I_log_transformed(i,j) = floor(min(255, 40*log(double(1+I(i,j)))));
       end
   end
   figure, imshow(I_log_transformed/max(I_log_transformed(:)));
   imwrite(I_log_transformed/max(I_log_transformed(:)), 'q3 Log-Transform.bmp');
end

