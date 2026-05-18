import numpy as np
import shap

from feature_extraction import FEATURE_NAMES


_FEATURE_LEN = len(FEATURE_NAMES)



def _default_categories():
    # feature vector is: 13 email + 32 url + 8 sender = 53
    email_len = 9
    url_len = 25
    sender_len = 5

    email_text = range(0, email_len)
    url = range(email_len, email_len + url_len)
    email_address = range(email_len + url_len, email_len + url_len + sender_len)
    return {
        "email_text": email_text,
        "url": url,
    }


def _scale_one(scaler, features):
    # scaler expects 2D
    return scaler.transform([features])


def _pick_phishing_class_shap(shap_raw):
    """Normalize SHAP output to a (n_features,) array for the phishing/positive class."""
    # Common shap outputs:
    # - binary: list([n_samples x n_features]_neg, [n_samples x n_features]_pos)
    # - binary: ndarray (n_samples x n_features) (single output)
    # - multiclass-like: array shaped (n_classes, n_samples, n_features)
    if isinstance(shap_raw, list):
        if len(shap_raw) < 2:
            return shap_raw[0]
        # positive class index 1
        return shap_raw[1]

    arr = np.array(shap_raw)

    # If shape is (n_samples, n_features)
    if arr.ndim == 2:
        return arr

    # If shape is (n_classes, n_samples, n_features)
    if arr.ndim == 3 and arr.shape[0] >= 2:
        return arr[1]

    # Fallback: try best-effort (will likely raise downstream)
    return arr


def _topk_by_abs(values, idxs, top_k):

    items = []

    for i in idxs:

        v = float(values[i])
        items.append(
            (FEATURE_NAMES[i], v)
        )

    items.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return dict(items[:top_k])

def _perturbation_importance(model, scaler, features, categories, top_k=5, zero_value=0.0):
    """Fallback importance via one-at-a-time perturbation.

    Note: this is NOT SHAP; it approximates feature impact on predicted probability.
    """
    features = np.array(features, dtype=float)
    base_prob = float(model.predict_proba(_scale_one(scaler, features)[0:1])[0][1])

    impacts = np.zeros(len(FEATURE_NAMES), dtype=float)
    for i in range(len(FEATURE_NAMES)):
        original = features[i]
        features[i] = zero_value
        new_prob = float(model.predict_proba(_scale_one(scaler, features)[0:1])[0][1])
        impacts[i] = abs(base_prob - new_prob)
        features[i] = original

    # Normalize to [0..1] style so values are comparable-ish
    max_v = float(np.max(impacts))
    if max_v > 0:
        impacts = impacts / max_v

    out = {}
    for cat, idxs in categories.items():
        # keep sign as positive impact only; this is for UI, not strict SHAP
        cat_items = []
        for i in idxs:
            v = float(impacts[i])
            # Keep all items to prevent empty charts
            cat_items.append((FEATURE_NAMES[i], v))
        cat_items.sort(key=lambda x: abs(x[1]), reverse=True)
        out[cat] = dict(cat_items[:top_k])

    return out


class SHAPTopFeatures:
    def __init__(self, model, scaler, feature_names=None, categories=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or FEATURE_NAMES
        self.categories = categories or _default_categories()

        self._explainer = None

        # Pre-create explainer once for performance
        try:
            self._explainer = shap.TreeExplainer(self.model)
        except Exception:
            self._explainer = None

    def explain_top_features_by_category(self, features, top_k=5):
        """Return stable top feature dicts grouped by category."""
        features = np.array(features, dtype=float)

        # Validate length
        if len(features) != len(self.feature_names):
            # Do not hard-fail; return empty categories so UI stays alive.
            return {k: {} for k in self.categories.keys()}


        features_scaled = _scale_one(self.scaler, features)

        # SHAP path
        if self._explainer is not None:
            try:
                shap_raw = self._explainer.shap_values(features_scaled)
                shap_sel = _pick_phishing_class_shap(shap_raw)

                # Ensure we have (n_samples, n_features)
                shap_sel = np.array(shap_sel)
                if shap_sel.ndim == 1:
                    # (n_features,) -> treat as one sample
                    shap_values = shap_sel
                elif shap_sel.ndim == 2:
                    shap_values = shap_sel[0]
                else:
                    # Last resort: flatten to feature dimension
                    shap_values = shap_sel.reshape(-1)

                result = {
                    "email_text": _topk_by_abs(
                        shap_values,
                        self.categories["email_text"],
                        top_k
                    ),

                    "url": _topk_by_abs(
                        shap_values,
                        self.categories["url"],
                        top_k
                    )
                }

                print(result)

                # If everything empty, fallback
                if not any(result[cat] for cat in result):
                    return _perturbation_importance(
                        self.model,
                        self.scaler,
                        features,
                        self.categories,
                        top_k=top_k,
                    )

                return result

            except Exception:
                # Fallback
                pass

        # Fallback path
        return _perturbation_importance(
            self.model,
            self.scaler,
            features,
            self.categories,
            top_k=top_k,
        )

