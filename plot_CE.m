function [] = plot_CE(CE_array,CE_array_avg,CE_array_valid)


[numbatches,epochs] = size(CE_array);
CE_values = reshape(CE_array,[1 epochs*numbatches]);

[num_valid,epochs] = size(CE_array_valid);
CE_valid_values = reshape(CE_array_valid,[1 num_valid*epochs]);

% figure(1)
% plot(CE_values)

%% Validation Cross-Entropy error plot
ticks = [0:10:epochs] * num_valid;
hold on
figure(1)
plot(CE_valid_values,'b')
%plot(CE_valid_values,'y--')
xlabel('epoch'); ylabel('CE'); title('Cross-Entropy error')
set(gca,'XTick',ticks)
set(gca,'XTickLabel',[0:10:epochs] );

