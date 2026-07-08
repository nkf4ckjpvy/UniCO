from __future__ import annotations

import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from barbar import Bar

from ..evaluation import best_f1_threshold, evaluate_threshold
from ..models import UniCO


class UniCOTrainer:
    def __init__(self, args: Namespace, train_loader, test_loader, device: torch.device):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model = UniCO(
            table_input_channels=args.table_input_channels,
            table_feature_dim=args.table_feature_dim,
            num_features=args.num_features,
            window_size=args.window_size,
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_layers=args.lstm_layers,
            bidirectional=args.bidirectional,
            image_input_channels=args.image_input_channels,
            image_feature_dim=args.image_feature_dim,
            fusion_bottleneck_dim=args.fusion_bottleneck_dim,
            final_dim=args.final_dim,
            fusion_type=args.fusion_type,
            pretrained_image_encoder=args.pretrained_image_encoder,
            use_lstm=not args.no_lstm,
            use_image=not args.no_image,
        ).to(device)

        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_milestones, gamma=0.1)
        self.center = self._set_center()
        self.total_train_time = 0.0
        self.avg_epoch_train_time = 0.0

    def _forward_training_batch(self, table_data: torch.Tensor, image_data: torch.Tensor | None):
        table_feature = self.model.table_encoder(table_data)
        if self.args.no_image:
            final_feature = self.model.fc_final(table_feature)
            regularization_loss = torch.zeros(1, device=self.device)
            consistency_loss = torch.zeros(1, device=self.device)
            return final_feature, regularization_loss, consistency_loss

        image_feature = self.model.image_encoder(image_data.to(self.device))
        fused_feature, regularization_loss = self.model.fusion_layer(table_feature, image_feature)
        final_feature = self.model.fc_final(torch.cat([table_feature, fused_feature], dim=1))
        consistency_loss = F.mse_loss(self.model.test_projection(table_feature), fused_feature.detach())
        return final_feature, regularization_loss, consistency_loss

    def _set_center(self, eps: float = 0.1) -> torch.Tensor:
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for batch_index, batch in enumerate(self.train_loader):
                table_data, image_data, _ = batch
                table_data = table_data.to(self.device)
                features, _, _ = self._forward_training_batch(table_data, image_data)
                outputs.append(features)
                if self.args.limit_train_batches and batch_index + 1 >= self.args.limit_train_batches:
                    break
        center = torch.cat(outputs, dim=0).mean(dim=0)
        center[(center.abs() < eps) & (center < 0)] = -eps
        center[(center.abs() < eps) & (center > 0)] = eps
        return center.detach()

    def train(self) -> None:
        train_start = time.perf_counter()
        epoch_times = []
        for epoch in range(self.args.num_epochs):
            epoch_start = time.perf_counter()
            self.model.train()
            totals = {"loss": 0.0, "svdd": 0.0, "reg": 0.0, "consistency": 0.0}
            batches = 0

            for batch_index, batch in enumerate(Bar(self.train_loader)):
                table_data, image_data, _ = batch
                table_data = table_data.to(self.device)
                self.optimizer.zero_grad()

                final_feature, regularization_loss, consistency_loss = self._forward_training_batch(table_data, image_data)

                svdd_loss = torch.mean(torch.sum((final_feature - self.center.to(self.device)) ** 2, dim=1))
                loss = (
                    svdd_loss
                    + self.args.reg_weight * regularization_loss.squeeze()
                    + self.args.consistency_lambda * consistency_loss
                )
                loss.backward()
                self.optimizer.step()

                totals["loss"] += loss.item()
                totals["svdd"] += svdd_loss.item()
                totals["reg"] += regularization_loss.squeeze().item()
                totals["consistency"] += consistency_loss.item()
                batches += 1

                if self.args.limit_train_batches and batch_index + 1 >= self.args.limit_train_batches:
                    break

            self.scheduler.step()
            epoch_time = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time)
            reg_label = "kl" if self.args.fusion_type == "ib" else "reg"
            print(
                f"Epoch {epoch + 1}/{self.args.num_epochs} | "
                f"loss={totals['loss'] / max(batches, 1):.4f} | "
                f"svdd={totals['svdd'] / max(batches, 1):.4f} | "
                f"{reg_label}={self.args.reg_weight * totals['reg'] / max(batches, 1):.4f} | "
                f"proj={self.args.consistency_lambda * totals['consistency'] / max(batches, 1):.4f} | "
                f"time={epoch_time:.2f}s"
            )

        self.total_train_time = time.perf_counter() - train_start
        self.avg_epoch_train_time = float(np.mean(epoch_times)) if epoch_times else 0.0

    def _score_loader(self, loader, limit_batches: int = 0) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        scores = []
        labels = []
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if len(batch) == 3:
                    table_data, _, label = batch
                else:
                    table_data, label = batch
                table_data = table_data.to(self.device)
                final_feature, _ = self.model(table_data, image_data=None)
                score = torch.sum((final_feature - self.center.to(self.device)) ** 2, dim=1)
                scores.append(score.cpu())
                labels.append(label.cpu())
                if limit_batches and batch_index + 1 >= limit_batches:
                    break

        raw_scores = torch.cat(scores, dim=0).numpy()
        labels_array = torch.cat(labels, dim=0).numpy()
        return raw_scores, labels_array

    def score_train(self) -> tuple[np.ndarray, np.ndarray]:
        return self._score_loader(self.train_loader, self.args.limit_train_batches)

    def score(self) -> tuple[np.ndarray, np.ndarray]:
        return self._score_loader(self.test_loader, self.args.limit_test_batches)

    def evaluate(self, segments: list[tuple[int, int]]) -> dict[str, object]:
        train_scores, _ = self.score_train()
        test_scores, labels = self.score()
        threshold_percentile = getattr(self.args, "threshold_percentile", 0.95)
        if not 0.0 <= threshold_percentile <= 1.0:
            raise ValueError("threshold_percentile must be between 0 and 1")
        threshold = float(np.quantile(train_scores, threshold_percentile))
        use_window_segment_adjust = not self.args.no_point_adjust

        predictions, threshold_result = evaluate_threshold(
            test_scores,
            labels,
            threshold,
            segments=segments,
            use_point_adjust=use_window_segment_adjust,
        )
        _, best_f1_reference = best_f1_threshold(
            test_scores,
            labels,
            segments=segments,
            step=self.args.threshold_step,
            use_point_adjust=use_window_segment_adjust,
        )
        metrics = threshold_result.to_dict()
        metrics["threshold_percentile"] = float(threshold_percentile)
        metrics["threshold_source"] = "train_scores"
        metrics["threshold_source_windows"] = int(len(train_scores))
        metrics["best_f1_reference"] = best_f1_reference.to_dict()
        metrics["adjustment"] = "window_level_segments" if use_window_segment_adjust else "none"
        metrics["num_window_level_segments"] = int(len(segments))
        metrics["num_test_windows"] = int(len(labels))
        metrics["predicted_anomalies"] = int(predictions.sum())
        metrics["train_time_seconds"] = self.total_train_time
        metrics["avg_epoch_time_seconds"] = self.avg_epoch_train_time
        return {
            "metrics": metrics,
            "scores": test_scores,
            "train_scores": train_scores,
            "labels": labels,
            "predictions": predictions,
        }
