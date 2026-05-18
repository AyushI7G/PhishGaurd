from flask import Flask, request, jsonify, send_from_directory
import xgboost as xgb
import numpy as np
import re
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

from gmail_api import (
    build_gmail_service,
    list_unread_messages,
    get_message_payload,
    extract_email_body
)

from feature_extraction import (
    extract_email_text_features,
    extract_urls,
    extract_url_features,
    fuse_features,
)
from shap_explainer import SHAPTopFeatures

from settings import MODEL_PATHS, FEATURE_DIMENSIONS

app = Flask(__name__, static_folder='../frontend')

# =====================================================
# CORS
# =====================================================

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

xgb_model = xgb.XGBClassifier()

xgb_model.load_model(MODEL_PATHS['xgb'])

scaler = joblib.load(MODEL_PATHS['scaler'])

model = xgb_model

print("Model loaded successfully!")

shap_helper = SHAPTopFeatures(
    model=model,
    scaler=scaler
)

# =====================================================
# HELPERS
# =====================================================

def clean_email_text(text):
    text = re.sub(r'\\s+', ' ', str(text))
    return text.strip()


def safe_vector(arr, size):
    arr = np.array(arr, dtype=float).flatten()

    if len(arr) < size:
        arr = np.concatenate([arr, np.zeros(size - len(arr))])

    elif len(arr) > size:
        arr = arr[:size]

    return arr


def classify_phishing(features):
    features = np.array(features, dtype=float).reshape(1, -1)

    features_scaled = scaler.transform(features)

    prob = model.predict_proba(features_scaled)[0][1]

    return float(prob)

# =====================================================
# AI REASONING
# =====================================================

def generate_reasoning(
    email_text,
    classification,
    probability,
    detected_features
):

    confidence = probability * 100

    reasoning_parts = []

    if classification == 'Phishing':

        reasoning_parts.append(
            "⚠️ This email has been identified as a likely phishing attempt."
        )

    elif classification == 'Suspicious':

        reasoning_parts.append(
            "⚠️ This email contains suspicious behavioral indicators."
        )

    else:

        reasoning_parts.append(
            "✅ This content appears to be LEGITIMATE."
        )

    reasoning_parts.append(
        f"Confidence Score: {confidence:.1f}%"
    )

    reasoning_parts.append("")

    if classification == 'Phishing':

        reasoning_parts.append(
            "⚡ Urgent language patterns resemble phishing behavior."
        )

    if detected_features:

        reasoning_parts.append(
            "Detected Indicators:"
        )

        for feat in detected_features[:5]:

            reasoning_parts.append(
                f"- {feat}"
            )

    if classification == 'Phishing':

        reasoning_parts.append(
            "Urgency patterns and suspicious structures strongly resemble phishing behavior."
        )

    return "\n".join(reasoning_parts)


# =====================================================
# RISK SCORE CALCULATION
# =====================================================

def calculate_risk_score(
    probability,
    email_text,
    detected_features
):

    if probability < 0.5:

        score = probability * 40

    else:

        score = probability * 100

    lower_email = email_text.lower()

    danger_words = [
        'urgent',
        'verify',
        'password',
        'bank',
        'login',
        'click'
    ]

    matched = sum(
        1 for w in danger_words
        if w in lower_email
    )

    score += matched * 3

    if 'http://' in lower_email:

        score += 10

    uppercase_ratio = sum(
        1 for c in email_text if c.isupper()
    ) / max(len(email_text), 1)

    if uppercase_ratio > 0.25:

        score += 5

    score += len(detected_features) * 2

    return min(round(score, 1), 100)


# =====================================================
# ROUTES
# =====================================================

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok'
    })


# =====================================================
# EMAIL PREDICTION
# =====================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.json

        email_text = clean_email_text(
            data.get('email_text', '')
        )

        # ---------------------------------------------
        # BASIC VALIDATION
        # ---------------------------------------------

        if not email_text.strip():

            return jsonify({
                'error': 'Please enter email content.'
            }), 400

        # ---------------------------------------------
        # FEATURE EXTRACTION
        # ---------------------------------------------

        email_feats = safe_vector(
            extract_email_text_features(email_text),
            FEATURE_DIMENSIONS['email_text']
        )

        urls = extract_urls(email_text)

        url_feature_list = []

        for u in urls:

            try:

                url_feature_list.append(
                    safe_vector(
                        extract_url_features(u),
                        FEATURE_DIMENSIONS['url']
                    )
                )

            except:
                continue

        if url_feature_list:

            url_feats = np.mean(
                url_feature_list,
                axis=0
            )

        else:

            url_feats = np.zeros(
                FEATURE_DIMENSIONS['url']
            )

        email_addr_feats = np.zeros(
            FEATURE_DIMENSIONS['email_address']
        )

        fused = np.concatenate([
            email_feats,
            url_feats,
            email_addr_feats
        ])

        # ---------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------

        prob = classify_phishing(fused)

        raw_score = round(prob * 100, 1)

        # ---------------------------------------------
        # CLASSIFICATION LOGIC
        # ---------------------------------------------

        if raw_score >= 80:

            classification = 'Phishing'

        elif raw_score >= 65:

            classification = 'Suspicious'

        else:

            classification = 'Legitimate'

        # ---------------------------------------------
        # REASONING
        # ---------------------------------------------

        # ---------------------------------------------
        # AI ANALYSIS + SHAP STYLE FEATURES
        # ---------------------------------------------

        detected_features = []

        lower_email = email_text.lower()

        # Suspicious keyword detection
        suspicious_words = [
            'urgent',
            'verify',
            'password',
            'login',
            'bank',
            'click',
            'security',
            'account',
            'suspend',
            'payment'
        ]

        matched_words = [
            w for w in suspicious_words
            if w in lower_email
        ]

        if matched_words:

            detected_features.append(
                f"Suspicious keywords detected: {', '.join(matched_words[:5])}"
            )

        # URL detection
        if 'http://' in lower_email or 'https://' in lower_email:

            detected_features.append(
                "External URL detected inside email body."
            )

        # Uppercase detection
        uppercase_ratio = sum(
            1 for c in email_text if c.isupper()
        ) / max(len(email_text), 1)

        if uppercase_ratio > 0.25:

            detected_features.append(
                "High uppercase character usage detected."
            )

        # Exclamation mark detection
        if email_text.count('!') >= 3:

            detected_features.append(
                "Excessive exclamation marks detected."
            )

        # AI analysis generation
        reasoning = generate_reasoning(
            email_text,
            classification,
            prob,
            detected_features
        )

        # Fallback SHAP display
        if not detected_features:

            detected_features.append(
                "No major suspicious indicators detected."
            )

        
        # ---------------------------------------------
        # DISPLAY RISK SCORE
        # ---------------------------------------------

        display_score = calculate_risk_score(
            prob,
            email_text,
            detected_features
        )

        if display_score >= 85:

            risk_level = 'High'

        elif display_score >= 50:

            risk_level = 'Medium'

        else:

            risk_level = 'Low'

        # ---------------------------------------------
        # SHAP-STYLE FEATURE EXPLANATIONS
        # ---------------------------------------------

        top_by_cat = shap_helper.explain_top_features_by_category(
            fused
        )

        top_email_features = top_by_cat.get(
            "email_text",
            {}
        )

        top_url_features = top_by_cat.get(
            "url",
            {}
        )

        # ---------------------------------------------
        # RESPONSE
        # ---------------------------------------------

        return jsonify({

            'probability': prob,

            'classification': classification,

            'risk_score': display_score,

            'risk_level': risk_level,

            'confidence_label': f'{display_score:.1f}%',

            'top_email_text_features':
                top_email_features,

            'top_url_features':
                top_url_features,

            'reasoning': reasoning,

            'ensemble_breakdown': {

                'XGBoost':
                    round(float(prob), 3),

                'Risk Calibration':
                    round(display_score / 100, 3),

                'Feature Fusion':
                    round(min(
                        1.0,
                        len(top_email_features) / 5
                    ), 3)
            },

            'model_version': 'v2-stable'
        })

    except Exception as e:

        print("PREDICT ERROR:", str(e))

        return jsonify({
            'error': str(e)
        }), 500

# =====================================================
# URL PREDICTION
# =====================================================

@app.route('/predict_url', methods=['POST'])
def predict_url():

    try:

        data = request.json

        url = str(data.get('url', '')).strip()

        trusted_domains = [
            'google.com',
            'youtube.com',
            'microsoft.com',
            'github.com',
            'openai.com',
            'amazon.com',
            'paypal.com',
            'apple.com',
            'facebook.com',
            'instagram.com',
            'linkedin.com',
            'wikipedia.org'
        ]

        for domain in trusted_domains:

            if domain in url.lower():

                return jsonify({

                    'probability': 0.01,

                    'classification': 'Safe Website',

                    'risk_score': 1,

                    'confidence_label': '99%',

                    'top_url_features': {},

                    'reasoning':
                        'Trusted domain detected.',

                    'ensemble_breakdown': {},

                    'model_version': 'v2-stable'
                })

        if len(url) < 3:
            return jsonify({
                'error': 'Invalid URL'
            }), 400

        url_feats = safe_vector(
            extract_url_features(url),
            FEATURE_DIMENSIONS['url']
        )

        email_feats = np.zeros(
            FEATURE_DIMENSIONS['email_text']
        )

        email_addr_feats = np.zeros(
            FEATURE_DIMENSIONS['email_address']
        )

        fused = np.concatenate([
            email_feats,
            url_feats,
            email_addr_feats
        ])

        prob = classify_phishing(fused)

        classification = (
            'Phishing Website'
            if prob > 0.85
            else 'Safe Website'
        )

        risk_score = round(prob * 100, 1)

        top_by_cat = shap_helper.explain_top_features_by_category(
            fused
        )

        top_url_features = top_by_cat.get(
            "url",
            {}
        )

        return jsonify({
            'probability': prob,
            'classification': classification,
            'risk_score': risk_score,
            'confidence_label': (
                f'{(100 - risk_score):.1f}%'
                if classification == 'Safe Website'
                else f'{risk_score:.1f}%'
            ),

            'top_url_features': top_url_features,

            'reasoning': (
                'Suspicious URL patterns detected.'
                if classification == 'Phishing Website'
                else 'Website appears safe.'
            ),

            'ensemble_breakdown': {

                'XGBoost':
                    round(float(prob), 3),

                'URL Structure Analysis':
                    round(min(1.0, risk_score / 100), 3),

                'Feature Fusion':
                    round(min(
                        1.0,
                        len(top_url_features) / 5
                    ), 3)
            },
            'model_version': 'v2-stable'
        })

    except Exception as e:

        print("URL PREDICT ERROR:", str(e))

        return jsonify({
            'error': str(e)
        }), 500


# =====================================================
# START SERVER
# =====================================================

@app.route('/scan_gmail', methods=['GET'])
def scan_gmail():

    try:

        service = build_gmail_service(
            credentials_path='credentials.json'
        )

        messages = list_unread_messages(
            service,
            max_results=25
        )

        results = []

        for msg in messages:

            payload = get_message_payload(
                service,
                msg['id']
            )

            email_body = extract_email_body(payload)

            if not email_body:
                continue

            email_feats = safe_vector(
                extract_email_text_features(email_body),
                FEATURE_DIMENSIONS['email_text']
            )

            urls = extract_urls(email_body)

            url_feature_list = []

            for u in urls:

                try:

                    url_feature_list.append(
                        safe_vector(
                            extract_url_features(u),
                            FEATURE_DIMENSIONS['url']
                        )
                    )

                except:
                    continue

            if url_feature_list:

                url_feats = np.mean(
                    url_feature_list,
                    axis=0
                )

            else:

                url_feats = np.zeros(
                    FEATURE_DIMENSIONS['url']
                )

            email_addr_feats = np.zeros(
                FEATURE_DIMENSIONS['email_address']
            )

            fused = np.concatenate([
                email_feats,
                url_feats,
                email_addr_feats
            ])

            prob = classify_phishing(fused)

            if prob > 0.95:
                classification = 'Phishing'
                color = '#dc3545'

            elif prob > 0.9:
                classification = 'Suspicious'
                color = '#ffc107'

            else:
                classification = 'Legitimate'
                color = '#28a745'

            results.append({
                'snippet': (
                    email_body
                    .replace('\n', '<br>')
                    .replace('\r', '')
                    .strip()[:250]
                ),
                'classification': classification,
                'probability': round(prob * 100, 2),
                'color': color
            })

        # ============================================
        # BUILD HTML UI
        # ============================================

        html = """

        <html>

        <head>

        <title>PhishGuard Gmail Scan</title>

        <style>

        body{
            font-family:Arial;
            background:#f5f7fa;
            padding:30px;
            color:#1e293b;
        }

        h1{
            text-align:center;
            margin-bottom:40px;
            color:#111827;
        }

        .summary{
            display:flex;
            gap:20px;
            justify-content:center;
            margin-bottom:40px;
            flex-wrap:wrap;
        }

        .summary-card{
            background:white;
            padding:20px;
            border-radius:14px;
            box-shadow:0 2px 10px rgba(0,0,0,0.08);
            min-width:180px;
            text-align:center;
            font-weight:bold;
            font-size:18px;
        }

        .card{
            background:white;
            padding:22px;
            border-radius:16px;
            margin-bottom:20px;
            box-shadow:0 2px 10px rgba(0,0,0,0.08);
        }

        .badge{
            padding:8px 14px;
            border-radius:8px;
            color:white;
            font-weight:bold;
            display:inline-block;
            margin-bottom:14px;
        }

        .prob{
            margin-top:10px;
            font-size:18px;
            font-weight:bold;
        }

        .snippet{
            margin-top:15px;
            line-height:1.7;
            color:#374151;
            white-space:pre-wrap;
        }

        </style>

        </head>

        <body>

        <h1>📧 Gmail Inbox Scan Results</h1>

        """

        phishing_count = 0
        suspicious_count = 0
        legit_count = 0

        for r in results:

            if r['classification'] == 'Phishing':
                phishing_count += 1

            elif r['classification'] == 'Suspicious':
                suspicious_count += 1

            else:
                legit_count += 1

            html += f"""

            <div class="card">

                <div
                    class="badge"
                    style="background:{r['color']};"
                >
                    {r['classification']}
                </div>

                <div class="prob">
                    Risk Score:
                    {r['probability']}%
                </div>

                <div class="snippet">
                    {r['snippet']}
                </div>

            </div>

            """

        html = f"""

        <div style="
            display:flex;
            gap:20px;
            margin-bottom:40px;
            justify-content:center;
        ">

            <div class="card">
                ✅ Legitimate: {legit_count}
            </div>

            <div class="card">
                ⚠ Suspicious: {suspicious_count}
            </div>

            <div class="card">
                🚨 Phishing: {phishing_count}
            </div>

        </div>

        """ + html

        html += """

        </body>
        </html>

        """

        return html

    except Exception as e:

        print("GMAIL SCAN ERROR:", str(e))

        return f"""

        <h1>Gmail Scan Error</h1>

        <p>{str(e)}</p>

        """

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )