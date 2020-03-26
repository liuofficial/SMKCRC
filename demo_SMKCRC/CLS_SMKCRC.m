function CLS_SMKCRC
% 2016-03-27
clc;
n = 1;
[img, img_gt, nClass, rows, cols, bands] = load_datas(n);
nts = 6; its = 10;

switch n,
    case 1, type = 2; smkcrc_sig = 0.5; smkcrc_lam = 1e-3; eta = 0.7;
        nCs = [20 50 100 200 400 800];
end

sigs = [0.02 : 0.02 : 0.1];

smkcrc_time = zeros(2,1);
t_begin = tic;
labels = SuperPixelMultiScale(img, rows, cols, nCs, sigs);
smkcrc_time(1) = toc(t_begin);

OA = zeros(nts, its); AA = zeros(nts, its); KA = zeros(nts, its); 
RA = zeros(nClass, nts, its); PR = zeros(rows*cols, nts, its); TM = zeros(2, nts, its);
for nt = 3,
    for it = 1 : its,
        disp(['======' num2str(nt) ',' num2str(it) '========']);
        [train_idx, test_idx] = load_train_test(n, type, nt, it);
        [Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
        
        t_begin = tic;
        P0 = RBF(img(:,Train.idx), img, smkcrc_sig, 2000);
        tau = 1e-3;
        smkcrc_pred = RelaxedSuperPixelMultiScaleCompositeKernel(img, Train.idx, labels, smkcrc_sig, ...
            smkcrc_lam, tau, eta, P0, Train.lab);
        smkcrc_time(2) = toc(t_begin);
        smkcrc_acc = class_eval(smkcrc_pred(Test.idx), Test.lab);
        OA(nt,it) = smkcrc_acc.OA; AA(nt,it) = mean(smkcrc_acc.ratio); KA(nt,it) = smkcrc_acc.KA;
        RA(:,nt,it) = smkcrc_acc.ratio; PR(:,nt,it) = smkcrc_pred; TM(:,nt,it) = smkcrc_time; 
        disp([OA(nt,it) AA(nt,it) KA(nt,it)]);
    end
    disp([mean(OA(nt,:)) mean(AA(nt,:)) mean(KA(nt,:))]);
end

end

function pred = RelaxedSuperPixelMultiScaleCompositeKernel(img, trainidx, labels, sig, ...
    lam, tau, eta, P0, lab)
train_size = length(trainidx);
labels_size = size(labels,2); eta_labels_size = eta / labels_size;
F = (P0(:,trainidx) .* (1-eta) + (lam*(1-eta) + tau) .* eye(train_size)) \ eye(train_size);
G = F;
S = F .* (1-eta) * P0;
for i = 1 : labels_size,
    label = labels(:, i);
    jdxs = sparse(bsxfun(@eq, label, unique(label)'));
    jdxs = bsxfun(@times, jdxs, 1./sum(jdxs));
    im = img * jdxs;
    Ks = RBF(im, im, sig, 2000);
    clear im;
    if min(label) == 0, label = label + 1; end
    train_label = label(trainidx);
    P = Ks(train_label, label);
    F = (P(:,trainidx) .* eta_labels_size  + (lam*eta_labels_size + tau) * eye(train_size)) \ eye(train_size);
    G = G + F;
    S = S + eta_labels_size .* F * P;
    clear Ks;
end
S = (1/(labels_size+1)) .* get_coef_lab(lab) / (eye(train_size)-G.*(tau/(labels_size+1))) * S;
[~,pred] = max(S);
end

function labels = SuperPixelMultiScale(img, rows, cols, nCs, sigs)
lam = 0.5; dist = 0; pcs = 0;
if pcs > 0,
    img = myPCA(img,pcs)'*img;
    img = normalization(img, 0, 255)';
else
    img = normalization(img, 0, 255, 1)';
end
img = reshape(img, rows, cols, size(img,2));
labels = mex_MSERS(img,nCs,lam,sigs,dist);
end
