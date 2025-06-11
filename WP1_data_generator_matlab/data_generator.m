% Clean the screen

clear all;
close all;

% Sim parameters

Nfft  = 2048;
GI    = 1/8;
Ng    = Nfft*GI;
Nofdm = Nfft+Ng;
Nsym  = 10000;
Nps   = 8; % Pilot spacing, Numbers of pilots and data per OFDM symbol
Np    = Nfft/Nps;
Nd    = Nfft-Np; 
Nbps  = 2;
M     = 2^Nbps; % Number of bits per (modulated) symbol
mm = 1; % LS channel estimation


% Channel Generation


% Delay_samples = [25, 50, 75, 125];
% rayChan = comm.RayleighChannel('SampleRate',fs,'MaximumDopplerShift',DopplerShift,...
% 'PathDelays',[0 Delay_samples/fs],'AveragePathGains',[0, -3, 1, -3, -2], 'NormalizePathGains', 0, 'PathGainsOutputPort', 1);
% reset(rayChan)

%---------------------------------------------------------------------
%----------------------- Channel Generation---------------------------
%---------------------------------------------------------------------
fs = 10e6;

% Loopback channel-------------------------------------------------------
DopplerShift = 10; 

NUM = 1;
if NUM == 1
    Delay_samples =     [10,  22];%[25, 50];%
    Attenuation   = [0, -20, -25];
elseif NUM == 2
    Delay_samples =    [0.3,   4,   6,   8,  20,  22,  24,  27,  30];   
    Attenuation   = [0, -13, -25, -23, -31, -26, -31, -25, -33, -23];
elseif NUM == 3
    Delay_samples =    [0.11,   0.22,   0.33,   0.44,  0.55,  0.66,  0.77,  0.98,  1.86, 6.34, 6.78]*10;   
    Attenuation   = [0, -6.3,   -8.8,  -11.1,     -8,  -8.2,    -6, -10.9, -12.4,   -11,-16.3, -10.8];
end

Attenuation_lin= 10.^(Attenuation/20);
k=1/sqrt(sum(Attenuation_lin.^2));    % Energy normalization
Attenuation_lin=Attenuation_lin.*k; 
Attenuation= 20*log10(Attenuation_lin);



rayChan = comm.RayleighChannel('SampleRate',fs,'MaximumDopplerShift',DopplerShift,...
'PathDelays',[0 Delay_samples/fs],'AveragePathGains', Attenuation,...
'NormalizePathGains', 0,'PathGainsOutputPort', 1);%, 'InitialTime', 1000000);
reset(rayChan)
% Transmitted signal generaiton and channel convolution (Same for the whole simulation)

XLong = [];
HLong = [];
YLong = [];
HLSLong = [];
SNR = 10;
  for nsym = 1:Nsym
       
        Xp = 2*(randn(1,Np)>0)-1;    % Pilot sequence generation
        
        msgint=randi([0 1],Nbps*(Nfft-Np),1);    % bit generation
       
        %Modulate
        
        Data = qammod(msgint, M, 'Gray','UnitAveragePower',true,'InputType','bit');
        ip = 0;
        pilot_loc = [];
        for k=1:Nfft
            if mod(k,Nps)==1
                X(k) = Xp(floor(k/Nps)+1);
                pilot_loc = [pilot_loc k]; 
                ip = ip+1;
            else
                X(k) = Data(k-ip);
            end
        end
        
        XLong = horzcat(XLong,X.');
        
        x = ifft(X,Nfft);                            % IFFT
        xt = [x(Nfft-Ng+1:Nfft) x];                  % Add CP
        
         %%Channel convolution

        [y_channel,hh_ideal] = rayChan(xt.');
        
        
        %Retrieve the ideal channel
        
        hh_ideal_paths=mean(hh_ideal);
        hh_ideal_delay= round(rayChan.PathDelays*fs);
        
        for idx_path=1:1:length(hh_ideal_delay)
            h(hh_ideal_delay(idx_path)+1)=hh_ideal_paths(idx_path);
        end
       
        sig_pow = mean(y_channel.*conj(y_channel));
        
%         Path_delay= 6;
%         
%         h(1)=(randn+1i*randn)/sqrt(2);
%         h(Path_delay)= (randn+1i*randn)/sqrt(4);
%         
        H = fft(h,Nfft); 
        HLong = horzcat(HLong, H.');
        channel_length = length(h);   % True channel and its time-domain length
        H_power_dB = 10*log10(abs(H.*conj(H)));        % True channel power in dB
        
        
        
        rand('seed',1);
        randn('seed',1);
        nose = 0;
        
        %%Noise
        
        yt = awgn(y_channel.',SNR,'measured');

        %Receiver

        y = yt(Ng+1:Nofdm);                   % Remove CP
        Y = fft(y);                           % FFT

        YLong = horzcat(YLong, Y.');

        if mm==1
            k=1:Np; 
            LS_est(k) = Y(pilot_loc(k))./Xp(k);  % LS channel estimation
            H_est = interpolate(LS_est,pilot_loc,Nfft,'linear'); 
            %Low pass frequency filtering
            h_est = ifft(H_est);
            h_filt = zeros(1,length(h_est));
            h_filt(1:channel_length) = h_est(1:channel_length);
            H_filt=fft(h_filt);
            H_est_power_dB = 10*log10(abs(H_filt.*conj(H_filt)));

        else

        %AI Channel estimation

        end
        HLSLong = horzcat(HLSLong, H_filt.');
        MSE (nsym) = mean(abs(H-H_filt).^2);
            %Equalization

        Y_eq = Y./H_filt;

        %Remove the pilot carriers
        ip = 0;
        for k=1:Nfft
            if mod(k,Nps)==1, ip=ip+1;  
            else
                Data_extracted(k-ip)=Y_eq(k);  
            end
        end

        %Demodulation
        msg_detected = reshape(qamdemod(Data_extracted, M, 'Gray','UnitAveragePower',true,'OutputType', 'bit'),1, []);

        noiseVar = 10.^(-SNR/10);
      
        if rem(nsym,500)==0
            fprintf('Simulated symbols = %d\n',nsym);
        end
  end
BER  = nose./(length(msgint)*Nsym);

dataWindow = 14;
labelWindow = 14;
slim = true;
channel_size = 2048;

[hEncNoise_long, hEnc_long] = PreProc(HLSLong , HLong, dataWindow, labelWindow, slim, channel_size);

hEncNoise = hEncNoise_long(1:Nfft * GI, :, :);
hEnc      = hEnc_long(1:Nfft * GI, :, :);

% Define the output directory
output_dir = fullfile('../matlab_output/');

% Create the directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save the data
save(fullfile(output_dir, 'h_perfect.mat'), 'hEnc');
save(fullfile(output_dir, 'h_ls_estimation.mat'), 'hEncNoise');

 
