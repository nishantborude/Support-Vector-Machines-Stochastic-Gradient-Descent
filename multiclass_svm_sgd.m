%parameter initialization
max_epoch = 200;
eta_0 = 1;
eta_1 = 100;
C = 0.1;

%dataset
%n = length(trLb);
[d, n] = size(trD);
k = length(unique(trLb));
w = zeros(d,k);
label_to_num = containers.Map(unique(trLb), 1:length(unique(trLb)));
num_to_label = containers.Map(1:length(unique(trLb)), unique(trLb));

%train the data
[weights, loss] = train(trD, trLb, max_epoch, eta_0, eta_1, C, n, d, k, w, label_to_num);

train_acc = acc_score(weights, trD, trLb, num_to_label)

cval_acc = acc_score(weights, valD, valLb, num_to_label)
%{
if tstD
    predictions = test(weights, tstD, num_to_label);
    a1.Class = predictions';
    writetable(a1, 'sample_submission.csv');
end
%}

l2_norm = sum(sum(weights.^2),2);

%algorithm
function [w, tot_loss] = train(trD, trLb, max_epoch, eta_0, eta_1, C, n, d, k, w, dict)
    for epoch = 1:max_epoch
       eta = eta_0 /(eta_1 + epoch);
       range_i = randperm(n);
       loss = 0;
       for i = range_i
           %calculate the score
           score = w' * trD(:, i);
           yi = dict(trLb(i));
           tmp = score(yi);
           score(yi) = -inf;
           [val, yi_hat] = max(score);
           score(yi) = tmp;

           cond = score(yi_hat) - score(yi) + 1;
           grad = w/n;
           if cond > 0
               grad(:, yi) = grad(:, yi) - C * trD(:, i);
               grad(:, yi_hat) = grad(:, yi_hat) + C * trD(:, i);
           end  
           %w_yi = 0;
           %w_yi = w(:, yi)/n;

           %w_yi_hat = 0;
           %w_yi_hat = w(:, yi_hat)/n;
           %if cond > 0
               %w_yi_hat C * trD(:, i);
               %w_yi_hat = w_yi_hat + C * trD(:, i);
           %end
           %grad(:, yi) = grad(:, yi) + w_yi;
           %grad(:, yi) = w_yi;
           %grad(:, yi_hat) = w_yi_hat;
           %grad(:, yi_hat) = grad(:, yi_hat) + w_yi_hat;
           w = w - (eta * grad);

           %nsum = 0;
           %for l = 1:k
           %    nsum = nsum + norm(w(:,l),2)^2;
           %end

           score_prime = w' * trD(:, i);
           yi_prime = dict(trLb(i));
           tmp = score_prime(yi_prime);
           score_prime(yi_prime) = -inf;
           [val, yi_hat_prime] = max(score_prime);
           score_prime(yi_prime) = tmp;

           cond_prime = score_prime(yi_hat_prime) - score_prime(yi_prime) + 1;
           %clear sum

           %loss = loss + nsum/(2*n) + C * max(cond_prime, 0);

           loss = loss + sum(sum(w.^2),2)/(2*n) + C * max(cond_prime, 0);
           %loss = loss + nsum/(2*n) + C * max(cond, 0);
       end
       tot_loss(epoch) = loss;
    end
end

function acc = acc_score(weights, data, labels, dict)
    pred = weights' * data;
    acc = 0;
    for i = 1:length(labels)
        [val, npred(i)] = max(pred(:,i));
        npred(i) = dict(npred(i));
        if npred(i) == labels(i)
            acc = acc + 1;
        end
    end
    acc = acc * 100/length(labels);
end

function npred = test(weights, data, dict)
    pred = weights' * data;
    [d, n] = size(data);
    for i = 1:n
       [val, npred(i)] = max(pred(:,i));
        npred(i) = dict(npred(i)); 
    end
end
