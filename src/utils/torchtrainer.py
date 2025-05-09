# import logging
# import os
#
# import numpy as np
# import pandas as pd
# import torch
# from scipy.stats import skew, kurtosis
# from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler, QuantileTransformer, \
#     PowerTransformer, Normalizer, Binarizer, FunctionTransformer
# from torch import nn, optim
# from torch.utils.data import DataLoader, TensorDataset
#
# from datasets.robustvision.preprocessor import subject_wise_normalization, global_normalization, \
#     separate_features_and_targets
# from src.utils.data_utils import create_lstm_tensors_dataset, create_dataloaders_dataset
#
# logger = logging.getLogger(__name__)
#
# def build_trainer(model: nn.Module,
#                   optimizer_cfg: dict,
#                   scheduler_cfg: dict = None,
#                   loss_cfg: dict = None,
#                   early_stopping_cfg: dict = None,
#                   device: torch.device = None):
#     """
#     Factory to create a TorchTrainer instance configured like FOVALTrainer.
#     """
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     return TorchTrainer(
#         model=model.to(device),
#         optimizer_cfg=optimizer_cfg,
#         scheduler_cfg=scheduler_cfg,
#         loss_cfg=loss_cfg,
#         early_stopping_cfg=early_stopping_cfg,
#         device=device
#     )
#
# class TorchTrainer:
#     def __init__(self,
#                  model: nn.Module,
#                  optimizer_cfg: dict,
#                  scheduler_cfg: dict = None,
#                  loss_cfg: dict = None,
#                  early_stopping_cfg: dict = None,
#                  device: torch.device = None):
#         self.best_transformers = None
#         self.transformers = None
#         self.early_stopping_limit = 200
#         self.target_scaler = StandardScaler() # MinMaxScaler(feature_range=(0, 1000))
#         self.model = model
#         self.device = device
#         self.optimizer_cfg = optimizer_cfg
#         self.scheduler_cfg = scheduler_cfg or {}
#         self.loss_cfg = loss_cfg or {'type': 'smoothl1', 'beta': 0.75}
#         self.early_cfg = early_stopping_cfg or {'patience': 100, 'min_delta': 0.0}
#         self._build_criterion()
#         self.optimizer = optim.AdamW(
#             self.model.parameters(),
#             lr=self.optimizer_cfg.get('lr', 0.032710957748580696),
#             weight_decay=self.optimizer_cfg.get('weight_decay', 0.09068313284126414)
#         )
#         # Scheduler
#         if self.scheduler_cfg:
#             self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
#                 self.optimizer,
#                 T_max=self.scheduler_cfg.get('T_max', 100),
#                 eta_min=self.scheduler_cfg.get('eta_min', 0)
#             )
#         else:
#             self.scheduler = None
#         # Early stopping
#         self.patience = self.early_cfg.get('patience', 200)
#         self.min_delta = self.early_cfg.get('min_delta', 0.0)
#         self.best_val = float('inf')
#         self.wait = 0
#
#
#     def _build_criterion(self):
#         t = self.loss_cfg.get('type', 'smoothl1')
#         if t == 'smoothl1':
#             beta = self.loss_cfg.get('beta', 0.75)
#             self.criterion = nn.SmoothL1Loss(beta=beta)
#         elif t == 'mae':
#             self.criterion = nn.L1Loss()
#         elif t == 'mse':
#             self.criterion = nn.MSELoss()
#         else:
#             self.criterion = nn.SmoothL1Loss(beta=0.75)
#             raise ValueError(f"Unknown loss type: {t}")
#
#
#     def set_dataloaders(self, train_loader: DataLoader, val_loader: DataLoader):
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         logger.info(f"[Trainer] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")
#
#     def fit_old(self, epochs: int, save_path: str = None):
#         history = {'train_loss': [], 'val_loss': []}
#         for epoch in range(1, epochs + 1):
#             train_loss = self._train_epoch()
#             history['train_loss'].append(train_loss)
#             val_loss = self._validate_epoch()
#             history['val_loss'].append(val_loss)
#             # scheduler step
#             if self.scheduler is not None:
#                 self.scheduler.step()
#             # early stopping
#             if val_loss + self.min_delta < self.best_val:
#                 self.best_val = val_loss
#                 self.wait = 0
#                 if save_path:
#                     os.makedirs(save_path, exist_ok=True)
#                     torch.save(self.model.state_dict(), os.path.join(save_path, 'best_model.pth'))
#             else:
#                 self.wait += 1
#                 if self.wait >= self.patience:
#                     logger.info(f"Early stopping at epoch {epoch}")
#                     break
#         return history
#
#     def fit(self, epochs):
#
#         # 0) Fit target‐scaler on train only
#         if self.target_scaler is not None:
#             all_y = []
#             for _, y in self.train_loader:
#                 all_y.append(y.detach().cpu().numpy().reshape(-1,1))
#             all_y = np.vstack(all_y)           # <-- do this after the loop
#             self.target_scaler.fit(all_y)  # just .fit, not fit_transform
#
#         # now normal training/validation loop
#         best_val_mae = float('inf')
#         patience = 0
#         # os.makedirs(save_path, exist_ok=True)
#         for epoch in range(1, epochs+1):
#             self.train_epoch(epoch)
#             val_mae = self.validate_epoch(epoch)
#             # step the cosine annealing schedule
#
#             if self.scheduler is not None:
#                 self.scheduler.step()
#
#             # early stopping
#             if val_mae < best_val_mae:
#                 best_val_mae = val_mae
#                 patience = 0
#             else:
#                 patience += 1
#
#             if patience > self.early_stopping_limit:
#                 break
#
#         # restore best
#         # self.model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
#         return best_val_mae
#
#     def train_epoch(self, epoch):
#         self.model.train()
#         total_loss = 0.0
#         for x, y_cm in self.train_loader:  # y_cm is already in cm
#             x, y_cm = x.to(self.device), y_cm.to(self.device)
#             self.optimizer.zero_grad()
#             out_cm = self.model(x).squeeze()  # predict in cm
#             loss = self.criterion(out_cm, y_cm)
#             loss.backward()
#             self.optimizer.step()
#             total_loss += loss.item() * x.size(0)
#         avg = total_loss / len(self.train_loader.dataset)
#         logger.info(f"[Epoch {epoch}] train_loss={avg:.4f} cm")
#         return avg
#
#     def validate_epoch(self, epoch):
#         self.model.eval()
#         total_err = 0.0
#         n = 0
#         debug_pairs = []
#
#         with torch.no_grad():
#             for x, y_scaled in self.val_loader:
#                 x = x.to(self.device)
#
#                 # 1) predict in scaled space
#                 out_scaled = self.model(x).squeeze().cpu() # .numpy().reshape(-1,1)
#
#                 # 2) invert to cm
#                 # out_cm = self.target_scaler.inverse_transform(out_scaled).ravel()
#                 # y_cm   = self.target_scaler.inverse_transform(
#                 #               y_scaled.cpu().numpy().reshape(-1,1)
#                 #          ).ravel()
#
#                 # accumulate error
#                 total_err += np.abs(out_scaled - y_scaled).sum()
#                 n += len(y_scaled)
#
#                 # # peek first 5
#                 # if len(debug_pairs) < 5:
#                 #     debug_pairs += list(zip(out_cm.tolist(), y_cm.tolist()))
#
#         val_mae = total_err / n
#         logger.info(f"[Epoch {epoch}] val_mae={val_mae:.2f} cm")
#
#         # print your peeked samples
#         # for i,(p,t) in enumerate(debug_pairs):
#         #     logger.info(f"   sample {i}: pred={p:.2f} cm   target={t:.2f} cm")
#
#         return val_mae
#
#     #
#     # def get_fold_loaders_scaled(self, dataset, held_out_subj, batch_size):
#     #
#     #
#     #     # 1) pull raw windows
#     #     X = dataset.features_raw.numpy()  # (N, seq_len, F)
#     #     y = dataset.targets_raw.numpy()  # (N,)
#     #     subs = np.array(dataset.subject_ids)
#     #
#     #     # 2) boolean masks
#     #     train_mask = subs != held_out_subj
#     #     val_mask = subs == held_out_subj
#     #
#     #     X_tr, y_tr = X[train_mask], y[train_mask]
#     #     X_va, y_va = X[val_mask], y[val_mask]
#     #
#     #     # 3) fit feature‐scaler on train
#     #     Ntr, L, F = X_tr.shape
#     #     feat_scaler = RobustScaler().fit(X_tr.reshape(-1, F))
#     #     X_tr_s = feat_scaler.transform(X_tr.reshape(-1, F)).reshape(Ntr, L, F)
#     #     X_va_s = feat_scaler.transform(X_va.reshape(-1, F)).reshape(-1, L, F)
#     #
#     #     # 4) fit target‐scaler on train
#     #     tgt_scaler = self.target_scaler  # your MinMax or Standard scaler, already stored
#     #     if self.target_scaler is not None:
#     #         # fit & transform train‐targets, transform val‐targets
#     #         y_tr_s = tgt_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
#     #         y_va_s = tgt_scaler.transform(y_va.reshape(-1, 1)).ravel()
#     #
#     #     # 5) build loaders
#     #     tr_ds = TensorDataset(torch.from_numpy(X_tr_s).float(),
#     #                           torch.from_numpy(y_tr_s).float())
#     #     va_ds = TensorDataset(torch.from_numpy(X_va_s).float(),
#     #                           torch.from_numpy(y_va_s).float())
#     #     return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True),
#     #             DataLoader(va_ds, batch_size=batch_size, shuffle=False))
#
#     def get_fold_loaders_scaled2(self,
#                                 features_raw: torch.Tensor,
#                                 targets_raw: torch.Tensor,
#                                 subject_ids: list,
#                                 held_out_subj: str,
#                                 batch_size: int):
#         """
#         Builds per‐fold train & val loaders by:
#           1) masking out the held‐out subject,
#           2) fitting a RobustScaler on *train* features & transforming both,
#           3) fitting a MinMaxScaler on *train* targets (cm!) & transforming both,
#           4) returning DataLoaders.
#         """
#         # 1) turn to numpy
#         X = features_raw.numpy()  # shape (N, seq_len, F)
#         y = targets_raw.numpy().reshape(-1, 1)  # shape (N,1)
#         subs = np.array(subject_ids)
#
#         # 2) split masks
#         train_mask = (subs != held_out_subj)
#         val_mask = (subs == held_out_subj)
#
#         X_tr, y_tr = X[train_mask], y[train_mask]
#         X_va, y_va = X[val_mask], y[val_mask]
#
#         # 3) feature‐scaling
#         Ntr, L, F = X_tr.shape
#         feat_scaler = RobustScaler().fit(X_tr.reshape(-1, F))
#         X_tr_s = feat_scaler.transform(X_tr.reshape(-1, F)).reshape(Ntr, L, F)
#         X_va_s = feat_scaler.transform(X_va.reshape(-1, F)).reshape(-1, L, F)
#
#         # 4) target‐scaling (cm)
#         tgt_scaler = StandardScaler() # MinMaxScaler(feature_range=(0, 1000))
#         y_tr_s = tgt_scaler.fit_transform(y_tr).ravel()  # fit only on train
#         y_va_s = tgt_scaler.transform(y_va).ravel()
#
#         # 5) wrap as TensorDatasets
#         tr_ds = TensorDataset(
#             torch.from_numpy(X_tr_s).float(),
#             torch.from_numpy(y_tr_s).float()
#         )
#         va_ds = TensorDataset(
#             torch.from_numpy(X_va_s).float(),
#             torch.from_numpy(y_va_s).float()
#         )
#
#         logger.info(f" after scaling → TRAIN y  min={y_tr_s.min():.3f}, max={y_tr_s.max():.3f}")
#         logger.info(f" after scaling →   VAL y  min={y_va_s.min():.3f}, max={y_va_s.max():.3f}")
#
#         # 6) DataLoaders
#         return (
#             DataLoader(tr_ds, batch_size=batch_size, shuffle=True),
#             DataLoader(va_ds, batch_size=batch_size, shuffle=False)
#         )
# #
# #
# # # ———————————————————————————————————————————————
# #     # 4) This is your new prepare_loader:
# #     def prepare_loader(self, dataset, subject_index, batch_size, is_train=False):
# #         subs = [subject_index] if not isinstance(subject_index, list) else subject_index
# #         df = dataset.input_features[dataset.input_features['SubjectID'].isin(subs)].copy()
# #
# #         # here  we collect all samples that belong to one of the subjects in this dataset (see subs
# #
# #         # 1) create features + clean + global/subject normalize
# #         df = self.normalize_data(df)
# #
# #         # 2) train‐only vs val: learn transforms on train, reapply on val
# #         if is_train:
# #             df = self.calculate_transformations_for_features(df)
# #         else:
# #             df = self.apply_transformations_on_features(df)
# #
# #         # 3) fit/transform features with a RobustScaler
# #         feat_cols = [c for c in df.columns if c not in ("Gt_Depth","SubjectID")]
# #         if is_train:
# #             self.feature_scaler = RobustScaler()
# #             df[feat_cols] = self.feature_scaler.fit_transform(df[feat_cols])
# #         else:
# #             df[feat_cols] = self.feature_scaler.transform(df[feat_cols])
# #
# #         # 4) fit/transform the target
# #         df = self.scale_target(df, isTrain=is_train)
# #
# #         # 5) sliding windows → sequences → tensors
# #         seqs = self.create_sequences(df)
# #         X, y = separate_features_and_targets(seqs)
# #
# #         X_t, y_t = create_lstm_tensors_dataset(X, y, is_train)
# #         return create_dataloaders_dataset(X_t, y_t, batch_size=batch_size)
# #
