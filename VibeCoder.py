# vibecoder_dashboard_app.py
# --- VibeCoder: Python code quality analyzer w/ dashboard ---
# NOTE: this got pretty big, might need cleanup later...

import os, re, ast, pickle, requests, logging
from collections import defaultdict
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# sklearn bits
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# nltk setup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# logging... probably should configure this better
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vibecoder")

# ------------------ NLTK bootstrap ------------------
# I always forget which resources are needed, so just brute forcing
for resource in ("punkt", "vader_lexicon", "stopwords"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)


# ------------------ Analyzer Class ------------------
class CodeAnalyzer:
    """
    Analyzer for Python code w/ ML features bolted on.
    Honestly this is a frankenstein, might refactor later.
    """

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_cols = []
        self.is_ready = False

        # NLP helpers
        self.sent = SentimentIntensityAnalyzer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

        # boot the classifiers
        self._init_models()

    def _init_models(self):
        # Could tune these hyperparams later
        self.models["rf"] = RandomForestClassifier(
            n_estimators=120, max_depth=10, random_state=42
        )
        self.models["gb"] = GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.1, max_depth=8, random_state=42
        )

    # ---------- Feature extraction (kinda messy) ----------
    def extract_feats(self, code: str) -> dict:
        """grab metrics out of python code"""
        if not isinstance(code, str):
            return self._empty_feats()

        feats = {}

        lines = code.split("\n")
        non_empty = [ln for ln in lines if ln.strip()]
        feats["line_count"] = len(lines)
        feats["non_empty_lines"] = len(non_empty)
        feats["char_count"] = len(code)
        feats["word_count"] = len(re.findall(r"\w+", code))

        avg_line_len = np.mean([len(l) for l in non_empty]) if non_empty else 0
        feats["avg_line_length"] = float(avg_line_len)

        # some quick regex counts
        feats["function_count"] = len(re.findall(r"def\s+\w+", code))
        feats["class_count"] = len(re.findall(r"class\s+\w+", code))
        feats["import_count"] = len(re.findall(r"import\s+\w+|from\s+\w+", code))
        feats["comment_count"] = len(
            re.findall(r"#.*|\"\"\".*?\"\"\"", code, re.DOTALL)
        )
        feats["if_count"] = len(re.findall(r"\bif\b", code))
        feats["for_count"] = len(re.findall(r"\bfor\b", code))
        feats["while_count"] = len(re.findall(r"\bwhile\b", code))

        # complexity-ish
        feats["cyclomatic_complexity"] = self._calc_complexity(code)
        feats["nesting_depth"] = self._calc_nesting(code)

        # halstead / readability
        feats.update(self._halstead(code))
        feats["readability_score"] = self._readability(code)

        # sentiment (lol, for code?)
        s = self.sent.polarity_scores(code)
        feats["sentiment_compound"] = float(s.get("compound", 0))

        # variable name stats
        vars_found = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
        feats["unique_variable_count"] = len(set(vars_found))
        feats["avg_variable_length"] = (
            float(np.mean([len(v) for v in vars_found])) if vars_found else 0.0
        )

        # AST deep dive
        feats.update(self._ast_features(code))
        return feats

    def _empty_feats(self):
        # fallback if code parse fails
        base_keys = [
            "line_count",
            "non_empty_lines",
            "char_count",
            "word_count",
            "avg_line_length",
            "function_count",
            "class_count",
            "import_count",
            "comment_count",
            "if_count",
            "for_count",
            "while_count",
            "cyclomatic_complexity",
            "nesting_depth",
            "halstead_volume",
            "halstead_difficulty",
            "halstead_effort",
            "readability_score",
            "sentiment_compound",
            "unique_variable_count",
            "avg_variable_length",
            "ast_node_count",
            "ast_depth",
        ]
        return {k: 0 for k in base_keys}

    def _calc_complexity(self, code):
        # kinda fake cyclomatic complexity
        keywords = ["if", "elif", "else", "for", "while", "try", "except", "with"]
        count = 1 + sum(code.count(k) for k in keywords)
        logic_ops = len(re.findall(r"\band\b|\bor\b", code))
        return max(1.0, count + 0.5 * logic_ops)

    def _calc_nesting(self, code):
        depth = 0
        for line in code.split("\n"):
            if line.strip():
                d = (len(line) - len(line.lstrip())) // 4
                if d > depth:
                    depth = d
        return depth

    def _halstead(self, code):
        try:
            ops = re.findall(r"[+\-*/=<>!&|^%]", code)
            operands = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b", code)

            n1, N1 = len(set(ops)), len(ops)
            n2, N2 = len(set(operands)), len(operands)

            if n1 == 0 or n2 == 0:
                return {
                    "halstead_volume": 0.0,
                    "halstead_difficulty": 0.0,
                    "halstead_effort": 0.0,
                }

            vocab = n1 + n2
            length = N1 + N2
            vol = length * np.log2(vocab)
            diff = (n1 / 2) * (N2 / n2)
            effort = vol * diff

            return {
                "halstead_volume": vol,
                "halstead_difficulty": diff,
                "halstead_effort": effort,
            }
        except Exception as e:
            log.warning(f"Halstead calc failed: {e}")
            return {
                "halstead_volume": 0.0,
                "halstead_difficulty": 0.0,
                "halstead_effort": 0.0,
            }

    def _readability(self, code):
        lines = [l for l in code.split("\n") if l.strip()]
        avg_len = np.mean([len(l) for l in lines]) if lines else 0
        comment_lines = len([l for l in lines if l.strip().startswith("#")])
        comment_ratio = comment_lines / max(1, len(lines))

        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
        quality = (
            sum(1 for t in tokens if len(t) > 2) / max(1, len(tokens)) if tokens else 0
        )

        # arbitrary scoring function (could be nonsense)
        score = (
            (max(0, 100 - (avg_len - 50) * 1.2) * 0.4)
            + (comment_ratio * 100 * 0.3)
            + (quality * 100 * 0.3)
        )
        return float(min(100, max(0, score)))

    def _ast_features(self, code):
        try:
            tree = ast.parse(code)
            counts = defaultdict(int)
            max_depth = 0

            def walk(n, d=0):
                nonlocal max_depth
                max_depth = max(max_depth, d)
                counts[type(n).__name__] += 1
                for c in ast.iter_child_nodes(n):
                    walk(c, d + 1)

            walk(tree)
            return {"ast_node_count": sum(counts.values()), "ast_depth": max_depth}
        except Exception:
            return {"ast_node_count": 0, "ast_depth": 0}

    # ---------- Training / Prediction ----------
    def train(self, df: pd.DataFrame):
        drop_cols = ["code", "quality"]
        X = df[[c for c in df.columns if c not in drop_cols]].astype(float)
        y = df["quality"].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, model in self.models.items():
            model.fit(X_train_s, y_train)

        # ensemble — soft voting
        vote = VotingClassifier(
            estimators=[(n, m) for n, m in self.models.items()], voting="soft"
        )
        vote.fit(X_train_s, y_train)

        self.ensemble = vote
        self.scaler = scaler
        self.feature_cols = X.columns.tolist()
        self.is_ready = True

        preds = vote.predict(X_test_s)
        return {
            "accuracy": accuracy_score(y_test, preds),
            "report": classification_report(y_test, preds, output_dict=True),
        }

    def predict(self, code: str):
        feats = self.extract_feats(code)
        X = np.array([feats.get(c, 0) for c in self.feature_cols]).reshape(1, -1)
        Xs = self.scaler.transform(X)

        proba = self.ensemble.predict_proba(Xs)[0]
        pred = self.ensemble.predict(Xs)[0]

        return {
            "prediction": str(pred),
            "probabilities": dict(zip(self.ensemble.classes_, proba)),
            "features": feats,
        }

    # ---------- Model persistence ----------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "ensemble": self.ensemble,
                    "scaler": self.scaler,
                    "features": self.feature_cols,
                },
                f,
            )

    def load(self, path: str):
        data = pickle.load(open(path, "rb"))
        self.ensemble = data["ensemble"]
        self.scaler = data["scaler"]
        self.feature_cols = data["features"]
        self.is_ready = True


# ---------------- GitHub Repo Loader ----------------
def fetch_github_repo(repo_url, save_dir="code_dataset", max_files=100):
    os.makedirs(save_dir, exist_ok=True)
    grabbed = []

    api_url = repo_url.replace("github.com", "api.github.com/repos") + "/contents/"
    r = requests.get(api_url)
    if r.status_code != 200:
        st.error("Repo not reachable, maybe private?")
        return []

    for item in r.json():
        if item["type"] == "file" and item["name"].endswith(".py"):
            rf = requests.get(item["download_url"])
            if rf.status_code == 200:
                out_path = os.path.join(save_dir, item["name"])
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(rf.text)
                grabbed.append(out_path)
                if len(grabbed) >= max_files:
                    break

    st.success(f"Pulled {len(grabbed)} .py files")
    return grabbed


def build_dataset(files, analyzer):
    data = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            code = fh.read()

        feats = analyzer.extract_feats(code)
        feats["code"] = code

        # fake labels... replace with real quality labels someday
        if feats["cyclomatic_complexity"] < 5 and feats["readability_score"] > 70:
            label = "excellent"
        elif feats["cyclomatic_complexity"] < 10:
            label = "good"
        elif feats["cyclomatic_complexity"] < 15:
            label = "average"
        else:
            label = "poor"

        feats["quality"] = label
        data.append(feats)

    return pd.DataFrame(data)


# ----------------- Streamlit UI -----------------
def _inject_css():
    # custom styling
    st.markdown(
        """
    <style>
    .stApp{background:linear-gradient(180deg,#071226,#021021);color:#e6eef6}
    .main-card{background:linear-gradient(180deg,rgba(10,20,30,0.6),rgba(2,8,16,0.6));padding:20px;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,0.6);}
    .brand{font-weight:700;font-size:20px;}
    .muted{color:#9fb0c8;}
    .kpi{background:#041726;padding:12px;border-radius:10px}
    .btn-gradient{background:linear-gradient(90deg,#ff4d6d,#ff9a2e);color:white;border-radius:8px;padding:8px 12px}
    </style>
    """,
        unsafe_allow_html=True,
    )


def run_app():
    st.set_page_config(page_title="VibeCoder", layout="wide")
    _inject_css()

    if "analyzer" not in st.session_state:
        st.session_state["analyzer"] = CodeAnalyzer()
    analyzer = st.session_state["analyzer"]

    # Sidebar nav
    st.sidebar.markdown('<div class="brand">VibeCoder</div>', unsafe_allow_html=True)
    nav = st.sidebar.radio("Navigation", ["Analyze", "Train", "Features", "About"])
    model_path = st.sidebar.text_input("Model path", "vibecoder.pkl")

    if st.sidebar.button("Save"):
        analyzer.save(model_path)
        st.sidebar.success("Saved model")

    if st.sidebar.button("Load"):
        analyzer.load(model_path)
        st.sidebar.success("Loaded model")

    # main card wrapper
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    if nav == "Analyze":
        st.header("Analyze some Python code")
        code = st.text_area("", height=300)
        if st.button("Predict quality"):
            if not analyzer.is_ready:
                st.warning("Train the model first!")
            else:
                result = analyzer.predict(code)
                st.success(f"Prediction: {result['prediction']}")
                st.table(
                    pd.DataFrame(
                        list(result["probabilities"].items()),
                        columns=["quality", "prob"],
                    )
                )

                st.subheader("Features extracted")
                st.dataframe(
                    pd.DataFrame.from_dict(
                        result["features"], orient="index", columns=["val"]
                    )
                )

    elif nav == "Train":
        st.header("Train on GitHub repo")
        repo_url = st.text_input(
            "GitHub repo URL", "https://github.com/python/cpython"
        )
        max_files = st.slider("Max Python files", 50, 500, 100, step=50)

        if st.button("Fetch repo"):
            files = fetch_github_repo(repo_url, max_files=max_files)
            if files:
                st.session_state["repo_df"] = build_dataset(files, analyzer)

        if "repo_df" in st.session_state and st.button("Train now"):
            df = st.session_state["repo_df"]
            res = analyzer.train(df)
            st.success(f"Training done! Accuracy: {res['accuracy']:.3f}")
            st.dataframe(df.head(15))

    elif nav == "Features":
        st.header("Feature Dashboard")
        if not analyzer.is_ready:
            st.warning("Need to train first")
            return

        df = pd.DataFrame(
            [analyzer.extract_feats(c["code"]) for c in st.session_state.get("repo_df", [])]
        )
        if df.empty:
            st.info("No dataset loaded yet")
            return

        # some viz
        features = [
            "cyclomatic_complexity",
            "halstead_volume",
            "readability_score",
            "sentiment_compound",
        ]
        fig = go.Figure()
        for f in features:
            fig.add_trace(go.Box(y=df[f], name=f))
        fig.update_layout(title="Metric Distributions", yaxis_title="value")
        st.plotly_chart(fig, use_container_width=True)

        # one-off histogram
        st.subheader("Pick a feature to explore")
        feat_choice = st.selectbox(
            "Feature", [c for c in df.columns if df[c].dtype in [int, float]]
        )
        fig2 = px.histogram(df, x=feat_choice, nbins=25, title=f"{feat_choice} dist")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.header("About")
        st.markdown(
            "VibeCoder: AI-powered Python code analyzer. "
            "Hacky dark UI w/ GitHub training support. Still WIP."
        )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    run_app()
