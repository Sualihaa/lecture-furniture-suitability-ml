function [summary, coefficients, merged_data, percentiles] = main_ergonomic_analysiss()
    % Ergonomic Suitability Analysis (No Gender, No Table Height)
    % Publication-ready outputs: confusion matrix, forest plot, coefficient table.

    rng(42);

    %% Output directory
    output_dir = 'Global_Model_Outputs_precisionopt_bootstrap_c';
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end

    %% CONFIG
    TARGET_PRECISION = 0.75;
    SMOTE_RATIO      = 0.50;
    MAX_ITERS        = 1e4;
    ALPHA_VAL        = 0.5;
    NUM_BOOT         = 500;
    BOOT_RAND_SEED   = 12345;

    %% 1) LOAD DATA
    fprintf('Loading data...\n');
    student_data   = readtable('furnituredata.xlsx','Sheet','allParticipants','VariableNamingRule','preserve');
    furniture_data = readtable('furnituredata.xlsx','Sheet','furniture','VariableNamingRule','preserve');
    fprintf('Student data size: %s\n', mat2str(size(student_data)));
    fprintf('Furniture data size: %s\n', mat2str(size(furniture_data)));

    %% 2) COMPUTE PREDICTORS & SUITABILITY
    [predictors_percentile, percentile_ranges] = compute_percentile_based_mismatch(student_data, furniture_data);
    percentiles = percentile_ranges;
    suitability = compute_suitability_basic(student_data, furniture_data);
    merged_data = join(predictors_percentile, suitability, 'Keys', {'StudentID','FurnitureID'});
    

    %% 3) DEFINE FEATURES & TARGET (NO GENDER)
    mismatch_cols = {'SeatHeightMismatch','SeatDepthMismatch','SeatWidthMismatch',...
                     'ClearanceMismatch','BackrestMismatch'};
    Y_all = (merged_data.SeatHeightFit & merged_data.SeatWidthFit & ...
             merged_data.SeatDepthFit & merged_data.BackrestFit & ...
             merged_data.ClearanceFit);
    Y_all = double(Y_all);
    X_all = table2array(merged_data(:, mismatch_cols));

    % Clean NaNs
    mask_good = ~any(isnan(X_all),2) & ~isnan(Y_all);
    X_all = X_all(mask_good,:); Y_all = Y_all(mask_good);
    fprintf('After cleaning: %d rows. Positives=%d, Negatives=%d\n', ...
        size(Y_all,1), sum(Y_all==1), sum(Y_all==0));

    if numel(unique(Y_all)) == 1
        error('Only one class present. Exiting.');
    end

    %% 4) TRAIN/TEST SPLIT
    cv = cvpartition(Y_all,'HoldOut',0.2,'Stratify',true);
    train_idx = training(cv); test_idx = test(cv);
    y_train = Y_all(train_idx); y_test = Y_all(test_idx);
    X_train_orig = X_all(train_idx,:); X_test = X_all(test_idx,:);

    %% 5) STANDARDIZE + SMOTE
    [mu, sig] = compute_mu_sig(X_train_orig);
    X_train_orig_std = (X_train_orig - mu) ./ sig;
    X_test_std = (X_test - mu) ./ sig;
    train_data_std = [X_train_orig_std, y_train];
    balanced_data = mySMOTE(train_data_std, 5, SMOTE_RATIO);
    X_train_bal = balanced_data(:,1:end-1); y_train_bal = balanced_data(:,end);

    %% 6) FIT MODEL
    fprintf('Fitting elastic-net model...\n');
    [B_path, FitInfo] = lassoglm(X_train_bal, y_train_bal, 'binomial', ...
        'Alpha', ALPHA_VAL, 'CV', 5, 'MaxIter', MAX_ITERS);
    idx_lambda = FitInfo.IndexMinDeviance;
    lambda_opt = FitInfo.Lambda(idx_lambda);
    beta_full = [FitInfo.Intercept(idx_lambda); B_path(:,idx_lambda)];

    % Predict
    X_test_withIntercept = [ones(size(X_test_std,1),1), X_test_std];
    p_noG = sigmoid(X_test_withIntercept * beta_full);
    y_test = y_test(:); p_noG = p_noG(:);

    %% 7) METRICS & THRESHOLD
    [bestT, prec, rec, f1] = choose_threshold_for_precision(y_test, p_noG, TARGET_PRECISION, 0.5);
    pred = double(p_noG >= bestT);
    %% --- A) EXPORT VALIDATION CASES FOR EXCEL CHECK ---
ncheck = 10;  % number of test cases to export
ncheck = min(ncheck, numel(p_noG));

% Choose 10 random test-set rows
rng(999);
pick_local = randperm(numel(p_noG), ncheck);

% Build a validation table containing RAW INPUTS + MATLAB probability
Tcheck = table();

% Raw student inputs (from merged_data, test set rows)
% We need the same rows that produced X_test_std and p_noG.
% We can grab those from merged_data using test_idx and mask_good alignment.

% Rebuild the row indices back to merged_data:
idx_all = find(mask_good);     % rows kept after cleaning
idx_test = idx_all(test_idx);  % rows in merged_data that correspond to test set

rows = idx_test(pick_local);   % pick 10 of them

Tcheck = merged_data(rows, {'S_PH','S_HW','S_BPL','S_SCH','S_TT','F_SH','F_SW','F_SD','F_UEB','F_STC'});
Tcheck.MATLAB_Prob = p_noG(pick_local);
Tcheck.MATLAB_Class = double(Tcheck.MATLAB_Prob >= bestT);

% Save as CSV
outCSV = fullfile(output_dir, 'excel_validation_cases.csv');
writetable(Tcheck, outCSV);
fprintf('✅ Wrote validation cases to: %s\n', outCSV);

    [acc, ~, ~, ~, auc, conf_mat] = calculate_metrics(y_test, pred, p_noG);
    LL = log_likelihood(y_test, p_noG);
    fprintf('\n===== PERFORMANCE =====\n');
    fprintf('Acc=%.1f%% AUC=%.3f F1=%.3f Prec=%.3f Rec=%.3f Thr=%.3f\n',...
        acc*100, auc, f1, prec, rec, bestT);

    %% 8) BOOTSTRAP CIs
    fprintf('Running bootstrap for CIs...\n');
    rng(BOOT_RAND_SEED);
    n_train = size(X_train_orig_std,1); p_feat = size(X_train_orig_std,2);
    boot_coefs = zeros(NUM_BOOT, p_feat + 1);
    for b = 1:NUM_BOOT
        idx_boot = randsample(n_train, n_train, true);
        Xb = X_train_orig_std(idx_boot,:); yb = y_train(idx_boot);
        [Bb, Fitb] = lassoglm(Xb, yb, 'binomial', 'Alpha', ALPHA_VAL, ...
            'Lambda', lambda_opt, 'MaxIter', MAX_ITERS);
        boot_coefs(b,:) = [Fitb.Intercept; Bb];
    end

    %% 9) COEFFICIENTS TABLE
    feat_names = [{'Intercept'}, mismatch_cols];
    base = beta_full; ci_low = prctile(boot_coefs, 2.5, 1)'; ci_up = prctile(boot_coefs, 97.5, 1)';
    OR = exp(base); OR_low = exp(ci_low); OR_up = exp(ci_up);
    coefficients = table(feat_names', base, ci_low, ci_up, OR, OR_low, OR_up, ...
        'VariableNames',{'Predictor','Beta','Beta_Lower','Beta_Upper','OR','OR_Lower','OR_Upper'});

    %% 10) SAVE MODEL & RESULTS
    save(fullfile(output_dir,'global_elasticnet_noGender.mat'),'beta_full','mu','sig','mismatch_cols','bestT','lambda_opt','percentile_ranges');
    save(fullfile(output_dir,'percentile_ranges.mat'),'percentile_ranges');

    summary = table(acc, auc, f1, prec, rec, LL, bestT, ...
        'VariableNames',{'Acc','AUC','F1','Prec','Rec','LogL','Threshold'});

    % --- SAVE JOURNAL-READY OUTPUTS ---
    % Confusion matrix (grayscale, publication quality)
    save_confusion_journal(conf_mat, fullfile(output_dir,'cm_noGender.png'), 'No Gender Model');

    % Coefficient table for manuscript (CSV)
    tbl_for_paper = coefficients(2:end,:);  % exclude intercept
    tbl_for_paper.Beta = round(tbl_for_paper.Beta, 3);
    tbl_for_paper.Beta_Lower = round(tbl_for_paper.Beta_Lower, 3);
    tbl_for_paper.Beta_Upper = round(tbl_for_paper.Beta_Upper, 3);
    tbl_for_paper.OR = round(tbl_for_paper.OR, 3);
    tbl_for_paper.OR_Lower = round(tbl_for_paper.OR_Lower, 3);
    tbl_for_paper.OR_Upper = round(tbl_for_paper.OR_Upper, 3);
    writetable(tbl_for_paper, fullfile(output_dir, 'coefficients_for_manuscript.csv'));

    % Forest plot
    save_forest_plot(coefficients, fullfile(output_dir,'forest_plot.png'));

    % Summary
    writetable(summary, fullfile(output_dir,'global_summary.csv'));
    writetable(merged_data, fullfile(output_dir,'merged_suitability.xlsx'));
    export_model_to_excel('Global_Model_Outputs_precisionopt_bootstrap_c');
    fprintf('\n✅ All outputs saved to: %s\n', output_dir);
end

%% ================== HELPER FUNCTIONS ==================

function [mu, sig] = compute_mu_sig(X)
    mu = mean(X,1); sig = std(X,0,1); sig(sig==0) = 1;
end

function s = sigmoid(z); s = 1./(1+exp(-z)); end

function [acc, prec, rec, f1, auc, conf_mat] = calculate_metrics(y_true, y_pred, scores)
    y_true = y_true(:); y_pred = y_pred(:);
    if numel(y_true) ~= numel(y_pred)
        error('Size mismatch in metrics.');
    end
    conf_mat = confusionmat(y_true, y_pred);
    acc = sum(y_true == y_pred) / numel(y_true);
    if numel(unique(y_true)) > 1
        TP = sum((y_true==1) & (y_pred==1));
        FP = sum((y_true==0) & (y_pred==1));
        FN = sum((y_true==1) & (y_pred==0));
        prec = TP / max(TP+FP, 1);
        rec  = TP / max(TP+FN, 1);
        f1   = 2*(prec*rec) / max(prec+rec, 1e-9);
        if nargin>2 && ~isempty(scores)
            [~,~,~,auc] = perfcurve(y_true, scores, 1);
        else
            auc = nan;
        end
    else
        prec=nan; rec=nan; f1=nan; auc=nan;
    end
end

function [bestT, bestPrec, bestRec, bestF] = choose_threshold_for_precision(y_true, scores, targetPrec, beta_fallback)
    y_true = y_true(:); scores = scores(:);
    if numel(y_true) ~= numel(scores)
        error('y_true and scores size mismatch.');
    end
    ths = linspace(0,1,1001);
    bestF = -inf; bestT = 0.5; bestPrec = NaN; bestRec = NaN;
    for t = ths
        pr = double(scores >= t);
        [~,prec,rec,f1] = calculate_metrics(y_true, pr, scores);
        if prec >= targetPrec && f1 > bestF
            bestF = f1; bestT = t; bestPrec = prec; bestRec = rec;
        end
    end
    if isnan(bestPrec)
        beta = beta_fallback;
        bestScore = -inf;
        for t = ths
            pr = double(scores >= t);
            [~,prec,rec,~] = calculate_metrics(y_true, pr, scores);
            fbeta = (1+beta^2)*(prec*rec)/max(beta^2*prec + rec, 1e-9);
            if fbeta > bestScore
                bestScore = fbeta; bestT = t; bestPrec = prec; bestRec = rec; bestF = fbeta;
            end
        end
    end
end

function [mismatch_df, percentile_ranges] = compute_percentile_based_mismatch(student_data, furniture_data)
    n_students = height(student_data);
    n_furniture = height(furniture_data);
    
    % Compute population percentiles
    measures = {'Popliteal_Height','Hip_Width','Buttock_Popliteal_Length',...
                'Subscapular_Height','Thigh_Thickness'};
    percentile_ranges = struct();
    for i = 1:numel(measures)
        m = measures{i};
        data = student_data.(m);
        data = data(~isnan(data));
        percentile_ranges.(m) = struct(...
            'p5', prctile(data,5),...
            'p50', prctile(data,50),...
            'p95', prctile(data,95));
    end
    
    clip01 = @(x) max(0, min(1, x));
    all_data = cell(n_students*n_furniture,1);
    c = 1;
    
    for i = 1:n_students
        s = student_data(i,:);
        for j = 1:n_furniture
            f = furniture_data(j,:);
            
            % Seat Height Mismatch
            ph = s.Popliteal_Height;
            ph_rng = percentile_ranges.Popliteal_Height;
            ph_pct = clip01((ph - ph_rng.p5) / (ph_rng.p95 - ph_rng.p5 + 1e-9));
            ideal_sh = ph_rng.p5 * cosd(30) + ph_pct * (ph_rng.p95 * cosd(5) - ph_rng.p5 * cosd(30));
            mismatch_SH = abs(f.SH - ideal_sh) / (ideal_sh + 1e-9);
            
            % Seat Width Mismatch
            hw = s.Hip_Width;
            hw_rng = percentile_ranges.Hip_Width;
            hw_pct = clip01((hw - hw_rng.p5) / (hw_rng.p95 - hw_rng.p5 + 1e-9));
            ideal_sw = 1.10 * hw_rng.p5 + hw_pct * (1.30 * hw_rng.p95 - 1.10 * hw_rng.p5);
            mismatch_SW = abs(f.SW - ideal_sw) / (ideal_sw + 1e-9);
            
            % Seat Depth Mismatch
            bpl = s.Buttock_Popliteal_Length;
            bpl_rng = percentile_ranges.Buttock_Popliteal_Length;
            bpl_pct = clip01((bpl - bpl_rng.p5) / (bpl_rng.p95 - bpl_rng.p5 + 1e-9));
            ideal_sd = 0.80 * bpl_rng.p5 + bpl_pct * (0.95 * bpl_rng.p95 - 0.80 * bpl_rng.p5);
            mismatch_SD = abs(f.SD - ideal_sd) / (ideal_sd + 1e-9);
            
            % Clearance Mismatch
            tt = s.Thigh_Thickness;
            mismatch_STC = max(0, tt - f.STC) / (tt + 1e-9);
            
            % Backrest Mismatch
            sch = s.Subscapular_Height;
            mismatch_UEB = max(0, f.UEB - sch) / (sch + 1e-9);
            
            all_data{c} = struct(...
                'StudentID', i,...
                'FurnitureID', j,...
                'SeatHeightMismatch', mismatch_SH,...
                'SeatDepthMismatch', mismatch_SD,...
                'SeatWidthMismatch', mismatch_SW,...
                'ClearanceMismatch', mismatch_STC,...
                'BackrestMismatch', mismatch_UEB);
            c = c + 1;
        end
    end
    mismatch_df = struct2table([all_data{:}]);
end

% 
function suitability = compute_suitability_basic(student_data, furniture_data)
% Five binary criteria (NO Table Height) + Baseline assessment columns
% Uses safe field names with prefixes: S_ (student), F_ (furniture), BL_ (baseline)

    n_students  = height(student_data);
    n_furniture = height(furniture_data);

    results = cell(n_students*n_furniture,1);
    k = 1;

    for i = 1:n_students
        % ---- Student measurements ----
        PH  = student_data.Popliteal_Height(i);
        HW  = student_data.Hip_Width(i);
        BPL = student_data.Buttock_Popliteal_Length(i);
        SCH = student_data.Subscapular_Height(i);
        TT  = student_data.Thigh_Thickness(i);

        for j = 1:n_furniture
            % ---- Furniture dimensions ----
            SHf  = furniture_data.SH(j);
            SWf  = furniture_data.SW(j);
            SDf  = furniture_data.SD(j);
            UEBf = furniture_data.UEB(j);
            STCf = furniture_data.STC(j);

            % ---- Baseline ranges/thresholds ----
            SH_min = PH * cosd(30);
            SH_max = PH * cosd(5);
            SH_margin = min(SHf - SH_min, SH_max - SHf);

            SW_min = 1.10 * HW;
            SW_max = 1.30 * HW;
            SW_margin = min(SWf - SW_min, SW_max - SWf);

            SD_min = 0.80 * BPL;
            SD_max = 0.95 * BPL;
            SD_margin = min(SDf - SD_min, SD_max - SDf);

            BR_margin = SCH - UEBf;

            CL_min = TT + 0.02;
            CL_margin = STCf - CL_min;

            % ---- Binary fits ----
            seat_height_fit = (SH_margin >= 0);
            seat_width_fit  = (SW_margin >= 0);
            seat_depth_fit  = (SD_margin >= 0);
            backrest_fit    = (BR_margin >= 0);
            clearance_fit   = (CL_margin > 0);

            % ---- Build struct in pieces (prevents EOL parse errors) ----
            s = struct();
            s.StudentID = i;
            s.FurnitureID = j;

            % Raw student
            s.S_PH  = PH;
            s.S_HW  = HW;
            s.S_BPL = BPL;
            s.S_SCH = SCH;
            s.S_TT  = TT;

            % Raw furniture
            s.F_SH  = SHf;
            s.F_SW  = SWf;
            s.F_SD  = SDf;
            s.F_UEB = UEBf;
            s.F_STC = STCf;

            % Baseline ranges/thresholds
            s.BL_SH_min = SH_min;
            s.BL_SH_max = SH_max;
            s.BL_SW_min = SW_min;
            s.BL_SW_max = SW_max;
            s.BL_SD_min = SD_min;
            s.BL_SD_max = SD_max;
            s.BL_BR_max = SCH;
            s.BL_CL_min = CL_min;

            % Baseline margins
            s.BL_SH_margin = SH_margin;
            s.BL_SW_margin = SW_margin;
            s.BL_SD_margin = SD_margin;
            s.BL_BR_margin = BR_margin;
            s.BL_CL_margin = CL_margin;

            % Binary outcomes (names used in main)
            s.SeatHeightFit = double(seat_height_fit);
            s.SeatWidthFit  = double(seat_width_fit);
            s.SeatDepthFit  = double(seat_depth_fit);
            s.BackrestFit   = double(backrest_fit);
            s.ClearanceFit  = double(clearance_fit);

            results{k} = s;
            k = k + 1;
        end
    end

    suitability = struct2table([results{:}]);
end


function balanced_data = mySMOTE(allData, k, target_ratio)
    % mySMOTE: Synthetic oversampling for binary classification
    labels = allData(:, end);
    features = allData(:, 1:end-1);
    classes = unique(labels);
    class_counts = arrayfun(@(c) sum(labels == c), classes);
    [~, maj_idx] = max(class_counts);
    majority_class = classes(maj_idx);
    allData_smote = allData;
    
    for ii = 1:numel(classes)
        cur_class = classes(ii);
        if cur_class == majority_class
            continue;
        end
        
        cur_count = class_counts(ii);
        target_count = round(class_counts(maj_idx) * target_ratio);
        needed = target_count - cur_count;
        if needed <= 0
            continue;
        end

        class_mask = (labels == cur_class);
        class_data = allData(class_mask, :);
        class_features = class_data(:, 1:end-1);
        
        current_k = min(k, size(class_data, 1) - 1);
        if current_k < 1
            synth_data = repmat(class_data, needed, 1);
        else
            synth_data = [];
            samples_per_inst = floor(needed / size(class_data, 1));
            extra = mod(needed, size(class_data, 1));
            
            for jj = 1:size(class_data, 1)
                cf = class_features(jj, :);
                [idx, ~] = knnsearch(class_features, cf, 'K', current_k + 1);
                nbr_idx = idx(2:end);
                n_gen = samples_per_inst + (jj <= extra);
                if isempty(nbr_idx) || n_gen <= 0
                    continue;
                end
                pick = nbr_idx(randi(numel(nbr_idx), n_gen, 1));
                for kk = 1:n_gen
                    nf = class_features(pick(kk), :);
                    alpha = rand;
                    sf = cf + alpha .* (nf - cf);
                    synth_data = [synth_data; [sf, cur_class]];
                end
            end
        end
        
        allData_smote = [allData_smote; synth_data];
    end

    balanced_data = allData_smote(randperm(size(allData_smote, 1)), :);
end

function ll = log_likelihood(y_true, p)
    epsv = 1e-15; p = max(min(p,1-epsv), epsv);
    ll = sum(y_true.*log(p) + (1-y_true).*log(1-p));
end

function save_confusion_journal(C, out_png, titleStr)
    % Ensure C is 2x2
    if isempty(C) || size(C,1) ~= 2 || size(C,2) ~= 2
        warning('Confusion matrix must be 2x2. Skipping plot.');
        return;
    end

    % Define class names
    classNames = {'Not Suitable', 'Suitable'};
    
    % Create figure
    fig = figure('Visible','off', 'PaperPositionMode','auto');
    ax = axes('Parent', fig);
    
    % Use grayscale colormap
    imagesc(ax, C);
    colormap(ax, flipud(gray(256)));
    caxis(ax, [0, max(C(:))]);  % Scale color to actual counts
    
    % Add white text labels
    for i = 1:2
        for j = 1:2
            txt = num2str(C(i,j), '%d');
            text(ax, j-1, i-1, txt, ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontSize',16, ...
                'FontWeight','bold', ...
                'Color','w');  % White text for contrast
        end
    end
    
    % Set axis labels and ticks
    ax.XTick = [0 1];
    ax.YTick = [0 1];
    ax.XTickLabel = classNames;
    ax.YTickLabel = classNames;
    ax.XAxisLocation = 'top';
    xlabel(ax, 'Predicted Class', 'FontSize',14, 'FontWeight','bold');
    ylabel(ax, 'True Class', 'FontSize',14, 'FontWeight','bold');
    title(ax, ['Confusion Matrix – ' titleStr], 'FontSize',16, 'FontWeight','bold');
    
    % Improve layout
    ax.TickLabelInterpreter = 'none';
    ax.FontSize = 12;
    ax.Layer = 'top';
    box(ax, 'on');
    % grid(ax, 'on', 'Color', [0.8 0.8 0.8]);
    
    % Save high-res
    exportgraphics(fig, out_png, 'Resolution', 300);
    close(fig);
end
function save_forest_plot(coefficients, out_png)
    tbl = coefficients(2:end,:); predictors = tbl.Predictor;
    or = tbl.OR; or_low = tbl.OR_Lower; or_up = tbl.OR_Upper;
    figure('Visible','off'); y = 1:length(predictors);
    scatter(or, y, 60, 'filled', 'MarkerFaceColor', [0.2 0.4 0.6]); hold on;
    for i = 1:length(y)
        plot([or_low(i), or_up(i)], [y(i), y(i)], '-', 'LineWidth', 2, 'Color', [0.6 0.6 0.6]);
    end
    plot([1,1], [0.5, length(y)+0.5], '--k', 'LineWidth', 1.5);
    yticks(y); yticklabels(predictors);
    set(gca, 'FontSize', 12); grid on; box on;
    xlabel('Odds Ratio (95% CI)', 'FontSize',14); ylabel('Predictor', 'FontSize',14);
    title('Forest Plot: Odds Ratios for Ergonomic Suitability', 'FontSize',16,'FontWeight','bold');
    xlim([min(or_low)*0.8, max(or_up)*1.2]); set(gca,'XScale','log');
    exportgraphics(gcf, out_png, 'Resolution', 300); close(gcf);
end

function export_model_to_excel(output_dir, out_xlsx)
%EXPORT_MODEL_TO_EXCEL  Export trained model + percentile params to Excel.
%
% What it exports (from your saved .mat files):
%  - Model: feature order, betas (no intercept), mu, sigma
%  - Meta : intercept, threshold(bestT), lambda_opt
%  - Percentiles: p5/p50/p95 for measures (at least PH/HW/BPL; others included if present)
%
% Usage examples:
%   export_model_to_excel('Global_Model_Outputs_precisionopt_bootstrap_c');
%   export_model_to_excel('Global_Model_Outputs_precisionopt_bootstrap_c', 'Ergo_Params.xlsx');

    if nargin < 1 || isempty(output_dir)
        error('Provide output_dir where the .mat files were saved.');
    end
    if nargin < 2 || isempty(out_xlsx)
        out_xlsx = fullfile(output_dir, 'Excel_Params.xlsx');
    else
        % if user provided just a filename, save into output_dir
        [p,~,~] = fileparts(out_xlsx);
        if isempty(p)
            out_xlsx = fullfile(output_dir, out_xlsx);
        end
    end

    model_mat = fullfile(output_dir, 'global_elasticnet_noGender.mat');
    pct_mat   = fullfile(output_dir, 'percentile_ranges.mat');

    if ~exist(model_mat,'file')
        error('Cannot find model file: %s', model_mat);
    end

    S = load(model_mat);

    % Required variables for prediction
    required = {'beta_full','mu','sig','mismatch_cols','bestT'};
    for r = 1:numel(required)
        if ~isfield(S, required{r})
            error('Missing "%s" in %s', required{r}, model_mat);
        end
    end

    % Load percentile_ranges (either in same file or separate)
    if isfield(S,'percentile_ranges')
        percentile_ranges = S.percentile_ranges;
    else
        if ~exist(pct_mat,'file')
            error('Cannot find percentile file: %s (and percentile_ranges not inside model mat)', pct_mat);
        end
        P = load(pct_mat);
        if ~isfield(P,'percentile_ranges')
            error('percentile_ranges not found in %s', pct_mat);
        end
        percentile_ranges = P.percentile_ranges;
    end

    beta_full     = S.beta_full(:);                  % (p+1)x1
    mu            = S.mu(:)';                        % 1xp
    sig           = S.sig(:)';                       % 1xp
    mismatch_cols = S.mismatch_cols;                 % 1xp cellstr
    bestT         = S.bestT;
    lambda_opt    = NaN;
    if isfield(S,'lambda_opt'), lambda_opt = S.lambda_opt; end

    % Safety: ensure order is consistent
    p = numel(mismatch_cols);
    if numel(beta_full) ~= p+1
        error('beta_full length (%d) does not match mismatch_cols (%d) + intercept.', numel(beta_full), p);
    end
    if numel(mu) ~= p || numel(sig) ~= p
        error('mu/sig length must match #features (%d). Got mu=%d sig=%d.', p, numel(mu), numel(sig));
    end

    % --------------------------
    % Sheet 1: Model parameters
    % --------------------------
    T_model = table( ...
        mismatch_cols(:), ...
        beta_full(2:end), ...
        mu(:), ...
        sig(:), ...
        'VariableNames', {'Feature','Beta','Mu','Sigma'} );

    % --------------------------
    % Sheet 2: Meta parameters
    % --------------------------
    T_meta = table( ...
        beta_full(1), bestT, lambda_opt, ...
        'VariableNames', {'Intercept','Threshold_bestT','Lambda_opt'} );

    % --------------------------
    % Sheet 3: Percentiles
    % --------------------------
    % Flatten struct-of-structs to rows: Measure, p5, p50, p95 (+ any other fields if present)
    meas = fieldnames(percentile_ranges);
    % Collect all percentile fieldnames found across measures
    allFields = {};
    for i = 1:numel(meas)
        fns = fieldnames(percentile_ranges.(meas{i}));
        allFields = union(allFields, fns);
    end

    % Prioritize common ones
    preferred = {'p5','p50','p95','mean','std'};
    % Keep preferred first (if they exist), then the rest
    ordered = preferred(ismember(preferred, allFields));
    rest = setdiff(allFields, ordered, 'stable');
    ordered = [ordered, rest];

    % Build table
    T_pct = table(string(meas), 'VariableNames', {'Measure'});
    for f = 1:numel(ordered)
        col = ordered{f};
        vals = nan(numel(meas),1);
        for i = 1:numel(meas)
            if isfield(percentile_ranges.(meas{i}), col)
                vals(i) = percentile_ranges.(meas{i}).(col);
            end
        end
        % Make valid variable name in Excel
        vn = matlab.lang.makeValidName(col);
        T_pct.(vn) = vals;
    end

% --------------------------
% Sheet 4: Legend / How-to
% --------------------------
legendHeader = {'Item','Meaning'};
legendBody = {
    'Feature order','Excel must use the same order as Model sheet Feature column.';
    'Standardization','x_std = (x - Mu) / Sigma';
    'Linear score','z = Intercept + SUM(Beta_i * x_std_i)';
    'Probability','p = 1/(1+EXP(-z))';
    'Decision','Suitable if p >= Threshold_bestT';
    'Percentiles sheet','Contains population percentiles used to compute targets in your mismatch method.';
    'Units','Inputs in Excel must use the same units as MATLAB data (meters vs cm affects +0.02 rule).'
};
T_legend = cell2table(legendBody, 'VariableNames', legendHeader);

    % --------------------------
    % Write to Excel
    % --------------------------
    % If file is open in Excel, write may fail. We try to delete first.
    if exist(out_xlsx,'file')
        try
            delete(out_xlsx);
        catch
            % If delete fails (file open), still attempt to write (may overwrite or error)
        end
    end

    writetable(T_model,  out_xlsx, 'Sheet','Model');
    writetable(T_meta,   out_xlsx, 'Sheet','Meta');
    writetable(T_pct,    out_xlsx, 'Sheet','Percentiles');
    writetable(T_legend, out_xlsx, 'Sheet','Legend');

    fprintf('✅ Exported Excel parameters to: %s\n', out_xlsx);
    fprintf('   Sheets: Model, Meta, Percentiles, Legend\n');
end
