%clear all;

%load('30realH2048IDL2.mat')
function y = domain_change(x, to_time)
    shape = size(x);

    y = zeros(shape);
    if to_time 
        for i= 1:shape(2)
            Htemp = x(:,i);
            y(:,i) = ifft(Htemp);
        end
    else
        for i= 1:shape(2)
            Htemp = x(:,i);
            y(:,i) = fft(Htemp);
        end
    end
    
end