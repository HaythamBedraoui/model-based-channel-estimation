% X, Y: Two arrays of the same size. In frequency. Symbols appended
% horizontally

%dataWindow: Horizontal size of the "picture".
%labelWindow: Horizontal size of the output (preferably 1)

%slim: if False the previous Windows should have the same size



function [hEncNoise, hEnc] = PreProc(HWide,HWideLOOPnorm, dataWindow, labelWindow, slim, chan_size)
    x = domain_change(HWide, true);
    y = domain_change(HWideLOOPnorm, true);
    
    x = x(1:chan_size, :);
    y = y(1:chan_size, :);
    if slim
        [xEnc] = encapsulate(x, dataWindow);
        [yEnc] = encapsulate(y, labelWindow);
        %yEnc = yEnc(:, :, dataWindow/2:length(yEnc)-dataWindow/2);
    else
        [xEnc, yEnc] = encapsulate(x, y, dataWindow);

    end
    hEncNoise = xEnc;
    hEnc = yEnc; 