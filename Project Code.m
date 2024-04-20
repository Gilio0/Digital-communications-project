close all;
clear all;
clc;
A=4;
Number_of_waveforms = 500;
Number_of_bits = 100;
% Initialize a [Number_of_waveforms X Number_of_bits] empty matrix
polar_NRZ_Ensemble = zeros(Number_of_waveforms,(Number_of_bits+1)*7);
polar_RZ_Ensemble = zeros(Number_of_waveforms,(Number_of_bits+1)*7);
unipolar_Ensemble = zeros(Number_of_waveforms,(Number_of_bits+1)*7);
% Generate random start indices based on delays
polar_NRZ_start_indices = randi([0, (Number_of_bits - 1)], 1, Number_of_waveforms);
polar_RZ_start_indices = randi([0, (Number_of_bits - 1)], 1, Number_of_waveforms);
unipolar_start_indices = randi([0, (Number_of_bits - 1)], 1, Number_of_waveforms);
for i= 1:Number_of_waveforms
    
    polar_NRZ_start_index = polar_NRZ_start_indices(i);
    polar_RZ_start_index = polar_RZ_start_indices(i);
    unipolar_start_index = unipolar_start_indices(i);
    polar_NRZ_Waveform = randi([0, 1], 1, Number_of_bits+1); % Generate a random vector of 100 elements where each element is either 0 or 1
    polar_RZ_Waveform = randi([0, 1], 1, Number_of_bits+1);
    unipolar_Waveform = randi([0, 1], 1, Number_of_bits+1);
    polar_NRZ_Tx1=((2*polar_NRZ_Waveform)-1)*A; % maping for 0 to be –A, 1 to be A
    polar_NRZ_Tx=repelem(polar_NRZ_Tx1,7);
    unipolar_Tx1=unipolar_Waveform*A; % maping for 0 to be 0, 1 to be A
    unipolar_Tx=repelem(unipolar_Tx1,7);
    polar_RZ_Tx1=((2*polar_RZ_Waveform)-1)*A; % maping for 0 to be –A, 1 to be A
    polar_RZ_Tx=repelem(polar_RZ_Tx1,7);
    for k=4:7:(Number_of_bits+1)*7
        polar_RZ_Tx(k+1:k+3) = 0;
    end
    
    % Shift the bits based on start index and delay
    polar_NRZ_shifted_bits = circshift(polar_NRZ_Tx, [0, -polar_NRZ_start_index]);
    polar_RZ_shifted_bits = circshift(polar_RZ_Tx, [0, -polar_RZ_start_index]);
    unipolar_shifted_bits = circshift(unipolar_Tx, [0, -unipolar_start_index]);
    % Discard the additional one bit
    polar_NRZ_sample_bits = polar_NRZ_shifted_bits(1:end-7);
    polar_RZ_sample_bits = polar_RZ_shifted_bits(1:end-7);
    unipolar_sample_bits = unipolar_shifted_bits(1:end-7);
    % the above array contains 100 elements, corresponding to the levels that will be transmitted in every pulse
    polar_NRZ_Ensemble(i, :) = polar_NRZ_shifted_bits; % every element represents 10 ms
    polar_RZ_Ensemble(i, :) = polar_RZ_shifted_bits;
    unipolar_Ensemble(i, :) = unipolar_shifted_bits;
end
figure(1);
polar_RZ_plot_sequence = repelem(polar_RZ_shifted_bits,1,10); %the vector plot sequence is a vector where each bit is represented as 70ms
polar_NRZ_plot_sequence = repelem(polar_NRZ_shifted_bits,1,10);
unipolar_plot_sequence = repelem(unipolar_shifted_bits,1,10);
subplot(3,1,1);
stairs(polar_RZ_plot_sequence, 'k', 'LineWidth', 2);
xlim([0, (Number_of_bits+1)*70]);
title('polar RZ Waveform');
ylabel('Amplitude(V)');
xlabel('Time(ms)');
grid on;
subplot(3,1,2);
stairs(polar_NRZ_plot_sequence, 'k', 'LineWidth', 2);
xlim([0, (Number_of_bits+1)*70]);
title('polar NRZ Waveform');
ylabel('Amplitude(V)');
xlabel('Time(ms)');
grid on;
subplot(3,1,3);
stairs(unipolar_plot_sequence, 'k', 'LineWidth', 2);
xlim([0, (Number_of_bits+1)*70]);
title('unipolar Waveform');
ylabel('Amplitude(V)');
xlabel('Time(ms)');
grid on;
%% Statistical mean
polar_RZ_column_sum=zeros(1,700);
polar_NRZ_column_sum=zeros(1,700);
unipolar_column_sum=zeros(1,700);
polar_RZ_column_sum = sum(polar_RZ_Ensemble(:, 8));
polar_NRZ_column_sum = sum(polar_NRZ_Ensemble(:, 8));
unipolar_column_sum = sum(unipolar_Ensemble(:, 8));
for j= 9:(Number_of_bits+1)*7
    polar_RZ_column_sum = [polar_RZ_column_sum, sum(polar_RZ_Ensemble(:, j))];
    polar_NRZ_column_sum = [polar_NRZ_column_sum, sum(polar_NRZ_Ensemble(:, j))];
    unipolar_column_sum = [unipolar_column_sum, sum(unipolar_Ensemble(:, j))];
    
end
polar_RZ_statistical_mean = polar_RZ_column_sum/Number_of_waveforms; 
polar_NRZ_statistical_mean = polar_NRZ_column_sum/Number_of_waveforms;
unipolar_statistical_mean = unipolar_column_sum/Number_of_waveforms;
time = 1:Number_of_bits*7; % X-axis for plotting
figure(2);
subplot(3,1,1);
plot(time,polar_RZ_statistical_mean);
ylim([min(-4), max(4)]);
title('Polar RZ Statistical mean');
xlabel('Time (ms) (each sample corresponding to 10 ms)');
ylabel('Amplitude');
grid on;
subplot(3,1,2);
plot(time,polar_NRZ_statistical_mean);
ylim([min(-4), max(4)]);
title('Polar NRZ Statistical mean');
xlabel('Time (ms) (each sample corresponding to 10 ms)');
ylabel('Amplitude');
grid on;
subplot(3,1,3);
plot(time,unipolar_statistical_mean);
ylim([min(-4), max(4)]);
title('unipolar Statistical mean');
xlabel('Time (ms) (each sample corresponding to 10 ms)');
ylabel('Amplitude');
grid on;
%% Time mean
%preallocating time mean matrices
polar_RZ_time_mean=zeros(Number_of_waveforms,1);
polar_NRZ_time_mean=zeros(Number_of_waveforms,1);
unipolar_time_mean=zeros(Number_of_waveforms,1);
polar_RZ_time_sum=zeros(Number_of_waveforms,1);
polar_NRZ_time_sum=zeros(Number_of_waveforms,1);
unipolar_time_sum=zeros(Number_of_waveforms,1);
%calculating mean of every realization
for i=1:Number_of_waveforms
    for j=1:Number_of_bits*7
    polar_RZ_time_sum(i,1)=polar_RZ_time_sum(i,1) + polar_RZ_Ensemble(i,j);
    polar_NRZ_time_sum(i,1)=polar_NRZ_time_sum(i,1) + polar_NRZ_Ensemble(i,j);
    unipolar_time_sum(i,1)=unipolar_time_sum(i,1) + unipolar_Ensemble(i,j);
    end
    polar_RZ_time_mean(i,1)=polar_RZ_time_sum(i,1)/(Number_of_bits*7);
    polar_NRZ_time_mean(i,1)=polar_NRZ_time_sum(i,1)/(Number_of_bits*7);
    unipolar_time_mean(i,1)=unipolar_time_sum(i,1)/(Number_of_bits*7);

end
waveform = 1:Number_of_waveforms;
figure(3)
subplot(3,1,1);
plot(waveform,polar_RZ_time_mean);
ylim([min(-4), max(4)]);
title('Polar RZ Time mean');
xlabel('Waveform');
ylabel('Amplitude');
grid on;

subplot(3,1,2);
plot(waveform,polar_NRZ_time_mean);
ylim([min(-4), max(4)]);
title('Polar NRZ Time mean');
xlabel('Waveform');
ylabel('Amplitude');
grid on;

subplot(3,1,3);
plot(waveform,unipolar_time_mean);
ylim([min(-4), max(4)]);
title('unipolar Time mean');
xlabel('Waveform');
ylabel('Amplitude');
grid on;
%% Statistical Autocorrelation
polar_RZ_stat_autocorr = zeros(1,Number_of_bits*7);
polar_NRZ_stat_autocorr = zeros(1,Number_of_bits*7);
unipolar_stat_autocorr = zeros(1,Number_of_bits*7);

% Calculate autocorrelation for each realization
polar_RZ_Ensemble_without_delay_bit = polar_RZ_Ensemble(:, 8:end);
polar_NRZ_Ensemble_without_delay_bit = polar_NRZ_Ensemble(:, 8:end);
unipolar_Ensemble_without_delay_bit = unipolar_Ensemble(:, 8:end);
dimensions = size(polar_RZ_Ensemble_without_delay_bit);
center_index = (dimensions(2)/2)+1;

for lag = (1-center_index):(center_index-2)
    
    for i = 1:Number_of_waveforms
        polar_RZ_sample = polar_RZ_Ensemble_without_delay_bit(i, :);
        polar_NRZ_sample = polar_NRZ_Ensemble_without_delay_bit(i, :);
        unipolar_sample = unipolar_Ensemble_without_delay_bit(i, :);
        polar_RZ_stat_autocorr(lag+center_index) = polar_RZ_stat_autocorr(lag+center_index) + (polar_RZ_sample(center_index)*polar_RZ_sample(lag+center_index));
        polar_NRZ_stat_autocorr(lag+center_index) = polar_NRZ_stat_autocorr(lag+center_index) + (polar_NRZ_sample(center_index)*polar_NRZ_sample(lag+center_index));
        unipolar_stat_autocorr(lag+center_index) = unipolar_stat_autocorr(lag+center_index) + (unipolar_sample(center_index)*unipolar_sample(lag+center_index));
        
    end
    
end

% Calculate average autocorrelation across all realizations
polar_RZ_avg_autocorr = polar_RZ_stat_autocorr/Number_of_waveforms;
polar_NRZ_avg_autocorr = polar_NRZ_stat_autocorr/Number_of_waveforms;
unipolar_avg_autocorr = unipolar_stat_autocorr/Number_of_waveforms;
%getting the zero lag at index zero
polar_RZ_avg_autocorr = circshift(polar_RZ_avg_autocorr, center_index-2);
polar_NRZ_avg_autocorr = circshift(polar_NRZ_avg_autocorr, center_index-2);
unipolar_avg_autocorr = circshift(unipolar_avg_autocorr, center_index-2);
%Flipping R_tau to the -ve quad
polar_RZ_autocorrelation = [fliplr(polar_RZ_avg_autocorr(2:end)) polar_RZ_avg_autocorr];
polar_NRZ_autocorrelation = [fliplr(polar_NRZ_avg_autocorr(2:end)) polar_NRZ_avg_autocorr];
unipolar_autocorrelation = [fliplr(unipolar_avg_autocorr(2:end)) unipolar_avg_autocorr];
figure(4)
time = -((Number_of_bits*7)-1):(Number_of_bits*7)-1;
subplot(3,1,1);
plot (time, polar_RZ_autocorrelation);
xlim([-35 35]);
ylim([min(-10), max(15)]);
xlabel("time in samples");
ylabel("Autocorrelation");
grid on;
title ("Statistical Autocorrelation (polar RZ)");
subplot(3,1,2);
plot (time, polar_NRZ_autocorrelation);
xlim([-35 35]);
ylim([min(-10), max(25)]);
xlabel("time in samples");
ylabel("Autocorrelation");
grid on;
title ("Statistical Autocorrelation (polar NRZ)");
subplot(3,1,3);
plot (time, unipolar_autocorrelation);
xlim([-35 35]);
ylim([min(-2), max(10)]);
xlabel("time in samples");
ylabel("Autocorrelation");
grid on;
title ("Statistical Autocorrelation (unipolar)");
%% PSD
figure(5)
polar_RZ_PSD = abs(fftshift(fft(polar_RZ_autocorrelation))) / 1399;
polar_RZ_freq_resolution = 100 / length(polar_RZ_PSD); % Frequency resolution where sampling freq is 100
polar_RZ_freq_axis = (-50 + (0:length(polar_RZ_PSD)-1) * polar_RZ_freq_resolution); % Create frequency axis from -50 to 50
subplot(3,1,1);
plot(polar_RZ_freq_axis, polar_RZ_PSD);
xlabel("Frequency (Hz)");
ylabel("PSD");
grid on;
title ("Power Spectral Density (Polar RZ)");
polar_NRZ_PSD = abs(fftshift(fft(polar_NRZ_autocorrelation))) / 1399;
polar_NRZ_freq_resolution = 100 / length(polar_NRZ_PSD); % Frequency resolution where sampling freq is 100
polar_NRZ_freq_axis = (-50 + (0:length(polar_NRZ_PSD)-1) * polar_NRZ_freq_resolution); % Create frequency axis from -50 to 50
subplot(3,1,2);
plot(polar_NRZ_freq_axis, polar_NRZ_PSD);
xlabel("Frequency (Hz)");
ylabel("PSD");
grid on;
title ("Power Spectral Density (Polar NRZ)");
unipolar_PSD = abs(fftshift(fft(unipolar_autocorrelation))) / 1399;
unipolar_freq_resolution = 100 / length(unipolar_PSD); % Frequency resolution where sampling freq is 100
unipolar_freq_axis = (-50 + (0:length(unipolar_PSD)-1) * unipolar_freq_resolution); % Create frequency axis from -50 to 50
subplot(3,1,3);
plot(unipolar_freq_axis, unipolar_PSD);
xlabel("Frequency (Hz)");
ylabel("PSD");
grid on;
title ("Power Spectral Density (unipolar)");
%% Time Autocorrelation

% Preallocate array to store autocorrelation values
polar_RZ_autocorr_values = zeros(Number_of_bits*7, 1);
polar_NRZ_autocorr_values = zeros(Number_of_bits*7, 1);
unipolar_autocorr_values = zeros(Number_of_bits*7, 1);
% Calculate autocorrelation for the realization
time_axis = 0:Number_of_bits*7-1;
polar_RZ_centered_realization = polar_RZ_Ensemble(1, :);
polar_NRZ_centered_realization = polar_NRZ_Ensemble(1, :);
unipolar_centered_realization = unipolar_Ensemble(1, :);
for lag = time_axis
    polar_RZ_autocorr_values(lag+1) = sum(polar_RZ_centered_realization(1:end-lag) .* polar_RZ_centered_realization(1+lag:end)) / (Number_of_bits*7 - lag);
    polar_NRZ_autocorr_values(lag+1) = sum(polar_NRZ_centered_realization(1:end-lag) .* polar_NRZ_centered_realization(1+lag:end)) / (Number_of_bits*7 - lag);
    unipolar_autocorr_values(lag+1) = sum(unipolar_centered_realization(1:end-lag) .* unipolar_centered_realization(1+lag:end)) / (Number_of_bits*7 - lag);
end
% Plot time autocorrelation within one realization
figure(6)
time_axis = [-fliplr(time_axis(2:end)), time_axis];
polar_RZ_time_Autocorrelation = [flipud(polar_RZ_autocorr_values(2:end)); polar_RZ_autocorr_values];
polar_NRZ_time_Autocorrelation = [flipud(polar_NRZ_autocorr_values(2:end)); polar_NRZ_autocorr_values];
unipolar_time_Autocorrelation = [flipud(unipolar_autocorr_values(2:end)); unipolar_autocorr_values];
subplot(3,1,1);
plot(time_axis,polar_RZ_time_Autocorrelation);
xlim([-35 35]);
ylim([min(-2), max(10)]);
xlabel('time in samples');
ylabel('Autocorrelation');
grid on;
title('Time Autocorrelation within One Realization (polar RZ)');
subplot(3,1,2);
plot(time_axis,polar_NRZ_time_Autocorrelation);
xlim([-35 35]);
ylim([min(-5), max(20)]);
xlabel('time in samples');
ylabel('Autocorrelation');
grid on;
title('Time Autocorrelation within One Realization (polar NRZ)');
subplot(3,1,3);
plot(time_axis,unipolar_time_Autocorrelation);
xlim([-35 35]);
ylim([min(-2), max(10)]);
xlabel('time in samples');
ylabel('Autocorrelation');
grid on;
title('Time Autocorrelation within One Realization (unipolar)');