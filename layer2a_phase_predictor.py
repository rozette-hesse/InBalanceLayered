from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PHASES = ["Fertility", "Follicular", "Luteal", "Menstrual"]


@dataclass
class Layer2AConfig:
    target_col: str = "phase"
    group_col: str = "id"
    random_state: int = 42
    test_size: float = 0.2
    max_iter: int = 3000


class Layer2AFeatureBuilder:
    """
    Builds Layer 2A features from:
    - symptom columns
    - composite symptom scores
    - PCA columns
    - cervical mucus columns
    - interaction features
    """

    def __init__(
        self,
        mucus_type_col: str = "cervical_mucus_estimated_type_final",
        mucus_score_col: str = "cervical_mucus_fertility_score_final",
    ) -> None:
        self.mucus_type_col = mucus_type_col
        self.mucus_score_col = mucus_score_col

        self.base_feature_order = [
            "headaches",
            "cramps",
            "sorebreasts",
            "fatigue",
            "sleepissue",
            "moodswing",
            "stress",
            "foodcravings",
            "indigestion",
            "bloating",
            "PC1",
            "PC2",
            "PC3",
            "PC4",
            "symptom_burden_score",
            "pain_score",
            "recovery_score",
            "mood_score",
            "body_score",
            "digestive_score",
            "cramps__x__bloating",
            "sorebreasts__x__foodcravings",
            "fatigue__x__sleepissue",
            "stress__x__moodswing",
            "pain_score__x__body_score",
            "digestive_score__x__mood_score",
            "cervical_mucus_score",
            "mucus_fertile_flag",
        ]

    @staticmethod
    def _map_mucus_type(value: object) -> str:
        x = str(value).strip().lower()

        if x in {"nan", "none", "", "unknown"}:
            return "unknown"
        if "egg" in x or "stretch" in x:
            return "eggwhite"
        if "water" in x or "slippery" in x:
            return "watery"
        if "cream" in x or "lotion" in x:
            return "creamy"
        if "stick" in x or "tacky" in x or "thick" in x or "cloudy" in x:
            return "sticky"
        if "dry" in x:
            return "dry"
        return "unknown"

    @staticmethod
    def _safe_mean(df: pd.DataFrame, cols: List[str]) -> pd.Series:
        available = [c for c in cols if c in df.columns]
        if not available:
            return pd.Series(0.0, index=df.index)
        return df[available].mean(axis=1)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # ---------- Composite scores ----------
        out["symptom_burden_score"] = self._safe_mean(
            out,
            [
                "headaches",
                "cramps",
                "sorebreasts",
                "fatigue",
                "sleepissue",
                "moodswing",
                "stress",
                "foodcravings",
                "indigestion",
                "bloating",
            ],
        )

        out["pain_score"] = self._safe_mean(
            out, ["cramps", "headaches", "sorebreasts", "bloating"]
        )

        out["recovery_score"] = self._safe_mean(
            out, ["fatigue", "sleepissue", "stress"]
        )

        out["mood_score"] = self._safe_mean(
            out, ["moodswing", "stress", "foodcravings"]
        )

        out["body_score"] = self._safe_mean(
            out, ["cramps", "sorebreasts", "bloating", "indigestion"]
        )

        out["digestive_score"] = self._safe_mean(
            out, ["bloating", "indigestion", "foodcravings"]
        )

        # ---------- Mucus features ----------
        if self.mucus_type_col in out.columns:
            out["cervical_mucus_type_std"] = out[self.mucus_type_col].apply(
                self._map_mucus_type
            )
        else:
            out["cervical_mucus_type_std"] = "unknown"

        if self.mucus_score_col in out.columns:
            out["cervical_mucus_score"] = out[self.mucus_score_col].fillna(0)
        else:
            out["cervical_mucus_score"] = 0

        out["mucus_fertile_flag"] = out["cervical_mucus_type_std"].isin(
            ["watery", "eggwhite"]
        ).astype(int)

        # ---------- Interaction features ----------
        interaction_pairs = [
            ("cramps", "bloating"),
            ("sorebreasts", "foodcravings"),
            ("fatigue", "sleepissue"),
            ("stress", "moodswing"),
            ("pain_score", "body_score"),
            ("digestive_score", "mood_score"),
        ]

        for a, b in interaction_pairs:
            if a in out.columns and b in out.columns:
                out[f"{a}__x__{b}"] = out[a] * out[b]
            else:
                out[f"{a}__x__{b}"] = 0

        # ---------- Ensure all expected feature cols exist ----------
        for col in self.base_feature_order:
            if col not in out.columns:
                out[col] = 0

        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in self.base_feature_order if c in df.columns]


class Layer2APredictor:
    """
    Train and predict cycle phase from symptoms + mucus.
    """

    def __init__(
        self,
        config: Optional[Layer2AConfig] = None,
        feature_builder: Optional[Layer2AFeatureBuilder] = None,
    ) -> None:
        self.config = config or Layer2AConfig()
        self.feature_builder = feature_builder or Layer2AFeatureBuilder()

        self.pipeline: Optional[Pipeline] = None
        self.feature_columns_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None

    def _make_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=self.config.max_iter,
                        class_weight="balanced",
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

    def prepare_training_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        df_feat = self.feature_builder.build_features(df)

        if self.config.target_col not in df_feat.columns:
            raise ValueError(f"Missing target column: {self.config.target_col}")
        if self.config.group_col not in df_feat.columns:
            raise ValueError(f"Missing group column: {self.config.group_col}")

        df_feat = df_feat[df_feat[self.config.target_col].notna()].copy()

        feature_cols = self.feature_builder.get_feature_columns(df_feat)
        X = df_feat[feature_cols].copy()
        y = df_feat[self.config.target_col].astype(str).copy()
        groups = df_feat[self.config.group_col].copy()

        return X, y, groups

    def fit(self, df: pd.DataFrame) -> "Layer2APredictor":
        X, y, _ = self.prepare_training_data(df)

        self.feature_columns_ = list(X.columns)
        self.pipeline = self._make_pipeline()
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["model"].classes_

        return self

    def fit_with_split(self, df: pd.DataFrame) -> Dict[str, object]:
        X, y, groups = self.prepare_training_data(df)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        self.feature_columns_ = list(X.columns)
        self.pipeline = self._make_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.classes_ = self.pipeline.named_steps["model"].classes_

        pred = self.pipeline.predict(X_test)
        proba = self.pipeline.predict_proba(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "classification_report": classification_report(y_test, pred),
            "confusion_matrix": pd.DataFrame(
                confusion_matrix(y_test, pred, labels=self.classes_),
                index=self.classes_,
                columns=self.classes_,
            ),
            "y_test": y_test,
            "pred": pred,
            "proba": pd.DataFrame(proba, columns=self.classes_, index=X_test.index),
            "X_test": X_test,
        }

        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None or self.feature_columns_ is None:
            raise RuntimeError("Model is not fitted yet.")

        df_feat = self.feature_builder.build_features(df)
        X = df_feat[self.feature_columns_].copy()

        pred = self.pipeline.predict(X)
        proba = self.pipeline.predict_proba(X)

        proba_df = pd.DataFrame(proba, columns=self.classes_, index=df.index)
        out = pd.DataFrame(index=df.index)
        out["predicted_phase"] = pred
        out["confidence"] = proba_df.max(axis=1)

        for cls in self.classes_:
            out[f"prob_{cls}"] = proba_df[cls]

        return out

    def get_coefficients(self) -> pd.DataFrame:
        if self.pipeline is None or self.feature_columns_ is None:
            raise RuntimeError("Model is not fitted yet.")

        model: BaseEstimator = self.pipeline.named_steps["model"]
        if not hasattr(model, "coef_"):
            raise RuntimeError("Underlying model does not expose coefficients.")

        coef_df = pd.DataFrame(
            model.coef_.T,
            index=self.feature_columns_,
            columns=model.classes_,
        )
        return coef_df.sort_index()

    def predict_one(self, row: Dict) -> Dict[str, object]:
        df = pd.DataFrame([row])
        pred_df = self.predict(df)
        result = pred_df.iloc[0].to_dict()
        return result


if __name__ == "__main__":
    # Example usage:
    # df = pd.read_csv("your_processed_layer2_dataset.csv")

    # predictor = Layer2APredictor()
    # metrics = predictor.fit_with_split(df)

    # print("Accuracy:", round(metrics["accuracy"], 4))
    # print("Balanced Accuracy:", round(metrics["balanced_accuracy"], 4))
    # print("Macro F1:", round(metrics["macro_f1"], 4))
    # print(metrics["classification_report"])
    # print(metrics["confusion_matrix"])

    # coef_df = predictor.get_coefficients()
    # print(coef_df)

    # predictor.fit(df)
    # sample_pred = predictor.predict(df.head(5))
    # print(sample_pred)

    pass
