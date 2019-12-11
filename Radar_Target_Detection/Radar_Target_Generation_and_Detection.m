clear all
clc;

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 100 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%

range_max = 200; 
range_resolution = 1;
speed_of_light = 3e8;
%% User Defined Range and Velocity of target
% Target's initial position and velocity. Note : Velocity
% remains constant

target_range = 50;
target_vel = 30;
 


%% FMCW Waveform Generation

% Design the FMCW waveform.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW
% chirp using the requirements above.
bandwidth = speed_of_light / (2 * range_resolution);
Tchirp = 5.5 * 2 * range_max / speed_of_light;
slope = bandwidth / Tchirp;

%Operating carrier frequency of Radar 
fc= 77e9;             %carrier freq

                                                          
%The number of chirps in one sequence. Its ideal to have 2^ value for the ease of running the FFT
%for Doppler Estimation. 
Nd=128;                   % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;                  %for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0,Nd*Tchirp,Nr*Nd); %total time for samples


%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx=zeros(1,length(t)); %transmitted signal
Rx=zeros(1,length(t)); %received signal
Mix = zeros(1,length(t)); %beat signal

%Similar vectors for range_covered and time delay.
r_t=zeros(1,length(t));
td=zeros(1,length(t));


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

for i=1:length(t)         
    
    %For each time stamp update the Range of the Target for constant velocity. 
    r_t(i) =  target_range + target_vel * t(i);

    %For each time sample we need to update the transmitted and
    %received signal.
    td(i) = 2*r_t(i)/speed_of_light;
    Tx(i) = cos(2*pi*(fc*t(i) + slope * 0.5 * t(i)^2));
    Rx (i)  = cos(2*pi*(fc*(t(i)-td(i)) + slope * 0.5 * (t(i)-td(i))^2));
    
    %Now by mixing the Transmit and Receive generate the beat signal
    %This is done by element wise matrix multiplication of Transmit and
    %Receiver Signal
    Mix(i) = Tx(i).*Rx(i);
    
end

%% RANGE MEASUREMENT

%reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
%Range and Doppler FFT respectively.
Mix  = reshape(Mix, [Nr, Nd]);

%run the FFT on the beat signal along the range bins dimension (Nr) and
%normalize.
signal_fft = fft(Mix,Nr)/Nr;

% Take the absolute value of FFT output
signal_fft = abs(signal_fft);


% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
P1  = signal_fft(1:Nr/2, :);


%plotting the range
figure ('Name','Range from First FFT')
subplot(2,1,1)

 % plot FFT output 
range_axis = linspace(0,200,Nr/2)*((Nr/2)/200);
plot(range_axis, P1(:,1))
axis ([0 200 0 1]);



%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

Mix=reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift (sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure,surf(doppler_axis,range_axis,RDM);

%% CFAR implementation

%Slide Window through the complete Range Doppler Map

%Select the number of Training Cells in both the dimensions.
Tr = 15;
Td = 10;

%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
Gr = 5;
Gd = 5;

% offset the threshold by SNR value in dB
offset = 6;


%Create a vector to store noise_level for each iteration on training cells
noise_level = zeros(1,1);

%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.


   % Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
   % CFAR
Nr = Nr/2;
RDM_new = RDM;

for i_cut = Tr+Gr+1 : Nr-(Tr+Gr) 
    for j_cut = Td+Gd+1: Nd-(Td+Gd)

        noise_level = zeros(1,1);
        
        %loop though the window
        for row = i_cut-(Tr+Gr):i_cut+Tr+Gr
            for col = j_cut-(Td+Gd):j_cut+Td+Gd
                
                %check if it is a training cell
                if(abs(i_cut-row) > Gr || abs(j_cut-col) > Gd)
                    noise_level = noise_level + db2pow(RDM(row,col));
                end
            end
        end

        number_of_training_cells = (2*Tr+2*Gr+1)*(2*Td+2*Gd+1) - (2*Gr+1)*(2*Gd+1);
        threshold = pow2db(noise_level / number_of_training_cells);
        
        threshold = threshold + offset;
        
        CUT=RDM(i_cut,j_cut);
        if (CUT > threshold)
            RDM_new(i_cut,j_cut) = 1;
        else
            RDM_new(i_cut,j_cut) = 0;
        end
    end     
end


% The process above will generate a thresholded block, which is smaller 
%than the Range Doppler Map as the CUT cannot be located at the edges of
%matrix. Hence,few cells will not be thresholded. To keep the map size same
% set those values to 0. 

edge_indices = find(RDM_new ~= 1);
RDM_new(edge_indices) = 0;


%display the CFAR output using the Surf function like we did for Range
%Doppler Response output.
figure,surf(doppler_axis,range_axis, RDM_new);
colorbar;


 
 