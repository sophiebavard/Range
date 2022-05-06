clear
%close all


%% Out-of-sample log-likelihood

% This scripts calculates the out-of-sample loglikelihood with the parameters optimized on half of the trials (saved form Model_fitting.m)
% Cross validation: the trials were optimized on half the learning phase (contexts 1 and 3) --> oos LL is calculated on the remaining half (contexts 2 and 4)
% Optimization: the the parameters were optimized on the learning phase (contexts 1:4) --> oss LL is calculated on the transfer phase (contexts 5:8)


%% SET UP

% cond 1 = 7.50 vs 2.50 learning test
% cond 2 = 7.50 vs 2.50 learning test
% cond 3 = 0.75 vs 0.25 learning test
% cond 4 = 0.75 vs 0.25 learning test
% cond 5 = 7.50 vs 0.75 transfer test
% cond 6 = 2.50 vs 0.25 transfer test
% cond 7 = 7.50 vs 0.25 transfer test
% cond 8 = 2.50 vs 0.75 transfer test

[sim, ok2] = listdlg('PromptString','Select a model',...
    'SelectionMode','single',...
    'ListString',{'cross validation','transfer simulation'}) ;

if sim == 1
    load('cross_validation');
elseif sim == 2
    load('Optimization');
end

n=800;
expe=experiment;

clear exp model
clear llCV
clear llTP

% MODELS
% 1 ABSOLUTE
% 2 RANGE
% 3 HABIT
% 4 UTILITY

%% Simulations

for sub = 1:numel(subjects)
    for model = 1:2

        params = parameters(sub,:,model);
        s = contexts{sub};
        a = choices{sub};
        R = outcomes{sub};
        C = coutcomes{sub};

        learning_fb(sub) = round((mod(expe{sub}(1)-1,4)+1)/2);
        transfer_fb(sub) = (mod(expe{sub}(1)-1,4)+1)*0*(mod(expe{sub}(1),2)==1)+(mod(expe{sub}(1)-1,4)+1)*1/2*(mod(expe{sub}(1),2)==0);

        % Parameters
        beta    = params(1); % choice temperature
        alphaQf = params(2); % alpha update Q-values
        alphaQc = params(3); % alpha update Q-values
        alphaV  = params(4); % alpha update context value
        n       = params(5); % curvature of the value function (0<m<1)

        % Initialization of the internal variables

        % Value matrices: 8 contexts with 2 options (index 1 = worst option, 2 = best option)
        Q  = zeros(8,2) ;   % Q-values (all models)
        H  = zeros(8,2) ;   % Habitual component (HABIT model)
        P  = zeros(8,2) ;   % Q-values to replace the arbiter (HABIT model)

        Rmin = zeros(8,1);  % Maximum value per context (RANGE model)
        Rmax = zeros(8,1);  % Minimum value per context (RANGE model)

        llCV(sub,model) = 0;            % out-of-sample log-likelihood cross validation
        llTP(sub,model) = 0;            % out-of-sample likelihood transfer phase

        r = zeros(length(s),1);     % simulated rewards
        c = zeros(length(s),1);     % simulated counterfactual rewards
        r2 = zeros(length(s),1);    % transformed rewards (RANGE and UTILITY)
        c2 = zeros(length(s),1);    % transformed counterfactual rewards (RANGE and UTILITY)

        for i = 1:length(s)

            if i == length(s)/2 +1 % first trial of post test

                % start the transfer phase with the final values of the learning phase (see Figure 1)
                Q(5,:) = [Q(3,2),Q(1,2)];
                Q(6,:) = [Q(3,1),Q(1,1)];
                Q(7,:) = [Q(4,1),Q(2,2)];
                Q(8,:) = [Q(4,2),Q(2,1)];

                H(5,:) = [H(3,2),H(1,2)];
                H(6,:) = [H(3,1),H(1,1)];
                H(7,:) = [H(4,1),H(2,2)];
                H(8,:) = [H(4,2),H(2,1)];

                P(5,:) = [P(3,2),P(1,2)];
                P(6,:) = [P(3,1),P(1,1)];
                P(7,:) = [P(4,1),P(2,2)];
                P(8,:) = [P(4,2),P(2,1)];

            end

            if sim == 1
                if s(i)==2 || s(i)==4 % sum loglikelihood for other conditions only

                    if ~isnan(a(i)), llCV(sub,model) = llCV(sub,model) + beta * Q(s(i),a(i)) - log(sum(exp(beta * Q(s(i),:)))); end

                end

            elseif sim == 2
                if s(i)>4 % sum loglikelihood for transfer phase only

                    if ~isnan(a(i)), llTP(sub,model) = llTP(sub,model) + log (exp(beta * Q(s(i),a(i))) / sum(exp(beta * Q(s(i),:)))); end

                end
            end


            if ~isnan(a(i))

                if s(i) < 5 % learning phase
                    r(i) = R(i)+9*(s(i)<3 & (R(i)==1)); % factual reward
                    c(i) = C(i)+9*(s(i)<3 & (C(i)==1)); % counterfactual reward
                elseif s(i) > 4 % transfer phase
                    if R(i)==1, r(i) = R(i)+(a(i)-1)*9; end % factual reward
                    if C(i)==1, c(i) = C(i)+(3-a(i)-1)*9; end % counterfactual reward
                end

                if model==1 % ABSOLUTE model (Qlearning)

                    if s(i) < 5 || transfer_fb(sub) ~= 0

                        deltaI(i) = r(i) - Q(s(i),a(i)) ;
                        Q(s(i),a(i)) =  Q(s(i),a(i))   + alphaQf * deltaI(i);

                        if learning_fb(sub) == 2 || transfer_fb(sub) == 2

                            deltaC(i) = c(i) - Q(s(i),3-a(i)) ;
                            Q(s(i),3-a(i)) =  Q(s(i),3-a(i))   + alphaQc * deltaC(i);

                        end
                    end

                elseif model==2 % RANGE model


                    if s(i) < 5 || transfer_fb(sub) ~= 0 % learning phase or transfer phase with feedback

                        if learning_fb(sub) == 2 || transfer_fb(sub) == 2 % complete feedback: take both outcomes into account

                            % updtate maximum
                            if max([r(i),c(i)]) > Rmax(s(i))
                                Rmax(s(i)) = Rmax(s(i)) + alphaV * (max([r(i),c(i)]) - Rmax(s(i)));
                            end

                            % update minimum (never done in our task design)
                            if min([r(i),c(i)]) < Rmin(s(i))
                                Rmin(s(i)) = Rmin(s(i)) + alphaV * (min([r(i),c(i)]) - Rmin(s(i)));
                            end

                        else % partial feedback: take only the factual reward into account

                            % updtate maximum
                            if r(i)> Rmax(s(i))
                                Rmax(s(i)) = Rmax(s(i)) + alphaV * (r(i) - Rmax(s(i)));
                            end

                            % update minimum (never done in our task design)
                            if r(i)< Rmin(s(i))
                                Rmin(s(i)) = Rmin(s(i)) + alphaV * (r(i) - Rmin(s(i)));
                            end
                        end


                        % normalized reward using range-adaptation
                        r2(i) = (r(i)-Rmin(s(i)))/(1+Rmax(s(i))-Rmin(s(i)));

                        % Q-value update with prediction error
                        deltaR(i) =  r2(i) - Q(s(i),a(i)) ;
                        Q(s(i),a(i))   = Q(s(i),a(i))   + alphaQf * deltaR(i);

                        if learning_fb(sub) == 2 || transfer_fb(sub) == 2 % complete feedback: update of the unchosen option

                            c2(i) = (c(i)-Rmin(s(i)))/(1+Rmax(s(i))-Rmin(s(i)));

                            deltaC(i) =  c2(i) - Q(s(i),3-a(i)) ;
                            Q(s(i),3-a(i))   = Q(s(i),3-a(i))   + alphaQc * deltaC(i);

                        end
                    end

                elseif model == 3 % HABIT model

                    % In this script, we note P the Qvalue matrix and Q the arbiter
                    % (noted D in the methods) because the softmax is performed
                    % with the matrix Q.
                    % Parameter alphaV represents habitual learning rate.
                    % Parameter n represents the weight w from the methods.

                    if s(i) < 5 || transfer_fb(sub) ~= 0 % learning phase or transfer phase with feedback

                        % Q-matrix update with prediction error
                        deltaR(i) = r(i) - P(s(i),a(i)) ;
                        P(s(i),a(i)) =  P(s(i),a(i))   + alphaQf * deltaR(i);

                        if learning_fb(sub) == 2 || transfer_fb(sub) == 2 % complete feedback: update of the unchosen option

                            deltaC(i) = c(i) - P(s(i),3-a(i)) ;
                            P(s(i),3-a(i)) =  P(s(i),3-a(i))   + alphaQc * deltaC(i);

                        end

                        % Habitual controller
                        if a(i) == 1 % choose option 1
                            a_t = [1 0];
                        elseif a(i) == 2 % choose option 2
                            a_t = [0 1];
                        end

                        % Habitual component
                        H(s(i),:)   = H(s(i),:) + alphaV * (a_t - H(s(i),:));

                        % Arbiter
                        Q(s(i),:)   = n * H(s(i),:)  +  (1-n) * P(s(i),:) ;

                    end


                elseif model == 4 % UTILITY model

                    if s(i) < 5 || transfer_fb(sub) ~= 0 % learning phase or transfer phase with feedback

                        % Utility reward
                        r2(i) = r(i)^n;

                        % Q-value update
                        deltaR(i) = r2(i) - Q(s(i),a(i)) ;
                        Q(s(i),a(i)) =  Q(s(i),a(i))   + alphaQf * deltaR(i);

                        if learning_fb(sub) == 2 || transfer_fb(sub) == 2  % complete feedback: update of the unchosen option

                            c2(i) = c(i)^n;

                            deltaC(i) = c2(i) - Q(s(i),3-a(i)) ;
                            Q(s(i),3-a(i)) =  Q(s(i),3-a(i))   + alphaQc * deltaC(i);

                        end
                    end
                end
            end
        end
    end
end





