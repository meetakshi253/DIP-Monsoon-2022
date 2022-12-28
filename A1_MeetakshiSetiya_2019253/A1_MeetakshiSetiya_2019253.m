%{
Name: Meetakshi Setiya
Roll no: 2019253 
DIP Assignment 1
%}

ques2b()
ques2c()

%Q2 b.
function [] = ques2b()
    fprintf("Q2 b.\nInput Matrix:\n")
    I = [2 0 0 0; 0 1 3 1; 3 0 2 0];
    I_out = bilinear_interpolate(I, 0.75); %c=0.75, given
    disp(I)
    fprintf("Matrix after interpolation:\n")
    disp(I_out)
end

%Q2 c.
function [] = ques2c()
    I = double(imread("x5.bmp"));
    I_out = bilinear_interpolate(I, 0.2); %c=0.2, given
    I_out = (I_out-min(I_out(:)))/max(I_out(:)); %normalize 
    figure('Name', 'Input Image'), imshow(uint8(I));
    figure('Name', 'Interpolated Image'), imshow(I_out)
    disp(size(I_out))
end

%--------------------------------------------------------%

function [out_img] = bilinear_interpolate(img, c)
    [rows, cols] = size(img);
    out_img = zeros(max(rows,floor(rows*c)),max(cols,floor(cols*c)));
    [out_rows, out_cols] = size(out_img);

    %now, interpolate each x,y for the output
    for i = 0:out_rows-1
        for j = 0:out_cols-1
            inp_i = i/c;
            inp_j = j/c;
            
            %co-ordinate outside matrix
            if inp_i>rows-1 && inp_j>cols-1
                out_img(i+1, j+1) = 0;

            %if point in the output coincides with any point in the input
            elseif floor(inp_i)==inp_i && floor(inp_j)==inp_j && (0<=inp_i) && (inp_i<=rows-1) && (0<=inp_j) && (inp_j<=cols-1)
                out_img(i+1,j+1) = img(inp_i+1, inp_j+1);

            %else, find 4 nearest neighbours and use bilinear interpolation
            else
                x1 = floor(inp_i);
                y1 = floor(inp_j);
                x2 = x1+1;
                y2 = y1+1;
                V = [0 0 0 0]; %zero padding by default. if valid pixel value exists, replace the 0.
                if (x1<=rows-1 && y1<=cols-1)
                    V(1) = img(x1+1, y1+1);
                end
                if (x1<=rows-1 && y2<=cols-1)
                    V(2) = img(x1+1, y2+1);
                end
                if (x2<=rows-1 && y1<=cols-1)
                    V(3) = img(x2+1, y1+1);
                end
                if (x2<=rows-1 && y2<=cols-1)
                    V(4) = img(x2+1, y2+1);
                end

                X = [x1, y1, x1*y1, 1; x1, y2, x1*y2, 1; x2, y1, x2*y1, 1; x2, y2, x2*y2, 1];
                A = X\V'; % inv(X)*V'
                out_v = A(1)*inp_i + A(2)*inp_j + A(3)*inp_i*inp_j + A(4);
                out_img(i+1,j+1) = out_v;
            end
        end
    end
end
