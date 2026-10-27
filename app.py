
import base64
from pathlib import Path

import pickle
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AutoPrice | Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. SESSION STATE INITIALIZATION
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""


# =========================================================
# 3. FILE PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "LinearRegressionModel.pkl"
DATA_PATH = BASE_DIR / "Cleaned car.csv"
IMAGE_PATH = BASE_DIR / "welcome_car.png"


# =========================================================
# 4. LOAD MODEL AND DATASET
# =========================================================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# =========================================================
# 5. CUSTOM CSS
# =========================================================
st.markdown("""
<style>
/* Main application */
.stApp {
    background: linear-gradient(135deg, #090b10, #171923);
    color: #ffffff;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3, p, label {
    color: #f5f5f5;
}

.hero {
    text-align: center;
    padding: 25px 10px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 17px;
    color: #b9bdc9;
}

.red {
    color: #ff414d;
}

/* Animated car track */
.car-track {
    position: relative;
    width: 100%;
    height: 160px;
    overflow: hidden;
    border-radius: 14px;
    border-bottom: 3px solid #ff414d;
    background: linear-gradient(180deg, #171923, #252630);
    margin: 20px 0 30px;
}

.car {
    position: absolute;
    bottom: 10px;
    left: -180px;
    font-size: 95px;
    line-height: 1;
    filter: drop-shadow(0 0 15px #ff2638);
    animation: drive 7s linear infinite;
}

@keyframes drive {
    0% {
        left: -180px;
        transform: translateY(0);
    }
    45% {
        transform: translateY(-5px);
    }
    50% {
        transform: translateY(0);
    }
    100% {
        left: calc(100% + 20px);
    }
}

/* Cards */
.info-card {
    background: #1c1f2a;
    border: 1px solid #343744;
    border-radius: 16px;
    padding: 22px;
    min-height: 145px;
    text-align: center;
    margin-bottom: 18px;
}

.info-card h3 {
    font-size: 20px;
}

.info-card p {
    color: #c2c5cf;
    font-size: 14px;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    background: #ff414d;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 45px;
    font-weight: 700;
    transition: 0.2s;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #e52f3c;
    color: white;
    border: none;
}

/* Form controls */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #242631;
    border-color: #414452;
}

/* Prediction result */
div[data-testid="stMetric"] {
    background: #20232d;
    border: 1px solid #3a3d49;
    border-radius: 14px;
    padding: 20px;
}

div[data-testid="stMetricValue"] {
    color: #ff515b;
}

/* Welcome screen */
.welcome-title {
    font-size: clamp(42px, 8vw, 90px);
    font-weight: 900;
    letter-spacing: 8px;
    text-align: center;
    color: white;
    text-shadow: 0 0 25px #ff2638;
    margin: 0;
}

.welcome-subtitle {
    text-align: center;
    font-size: 16px;
    letter-spacing: 4px;
    color: #ffffff;
    margin-top: 15px;
}

/* Divider */
hr {
    border-color: #363946;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 6. WELCOME PAGE
# Full-screen car image with WELCOME and Continue button
# =========================================================
if st.session_state.page == "welcome":

    if IMAGE_PATH.exists():
        image_bytes = IMAGE_PATH.read_bytes()
        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        image_background = (
            f"data:image/png;base64,{encoded_image}"
        )

        st.markdown(f"""
        <style>
        .block-container {{
            max-width: 100% !important;
            padding: 0 !important;
        }}

        .welcome-screen {{
            min-height: 100vh;
            width: 100%;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background:
                linear-gradient(
                    rgba(0,0,0,0.20),
                    rgba(0,0,0,0.30)
                ),
                url("{image_background}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            text-align: center;
        }}

        .welcome-screen h1 {{
            font-size: clamp(45px, 8vw, 95px);
            font-weight: 900;
            letter-spacing: 8px;
            color: white;
            text-shadow: 0 0 30px #ff2638;
            margin: 0;
        }}

        .welcome-screen p {{
            color: white;
            letter-spacing: 4px;
            font-size: 16px;
        }}
        </style>

        <div class="welcome-screen">
            <h1>WELCOME</h1>
            <p>YOUR JOURNEY STARTS HERE</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Fallback if the image file is missing
        st.markdown("""
        <div style="
            min-height:75vh;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            text-align:center;
            background:linear-gradient(135deg,#090b10,#47101a);
            border-radius:18px;
        ">
            <div style="font-size:100px;">🏎️</div>
            <h1 style="
                font-size:70px;
                letter-spacing:8px;
                text-shadow:0 0 25px #ff2638;
            ">WELCOME</h1>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "To use your welcome image, place "
            "'welcome_car.png' beside app.py."
        )

    _, button_col, _ = st.columns([1, 1, 1])

    with button_col:
        if st.button(
            "Continue →",
            key="welcome_continue",
            use_container_width=True
        ):
            st.session_state.page = "home"
            st.rerun()


# =========================================================
# 7. HOME PAGE
# =========================================================
elif st.session_state.page == "home":

    st.markdown("""
    <div class="hero">
        <h1>Auto<span class="red">Price</span></h1>
        <p>
            Discover your car's estimated market value.
        </p>
    </div>

    <div class="car-track">
        <div class="car">🏎️</div>
    </div>

    <div class="info-card">
        <h2>Find the right value for your next ride.</h2>
        <p>
            Enter your car details and get an estimated price
            powered by Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🚘 Car Details</h3>
            <p>Explore different brands and models.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>⚡ Quick Estimate</h3>
            <p>Get a prediction from your trained model.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>📊 ML Powered</h3>
            <p>Built with Python and Linear Regression.</p>
        </div>
        """, unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 1.5, 1])

    with center_col:
        if st.button(
            "Get Started →",
            key="get_started",
            use_container_width=True
        ):
            st.session_state.page = "login"
            st.rerun()


# =========================================================
# 8. LOGIN PAGE
# DEMO LOGIN ONLY - NOT SECURE AUTHENTICATION
# =========================================================
elif st.session_state.page == "login":

    _, form_col, _ = st.columns([1, 1.5, 1])

    with form_col:
        st.markdown("""
        <div class="hero">
            <h1 style="font-size:40px;">
                Welcome <span class="red">Back</span>
            </h1>
            <p>
                Sign in to access your car price predictor.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username")

            password = st.text_input(
                "Password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Login →",
                use_container_width=True
            )

            if submitted:
                if username.strip() and password:
                    # Demo only: any non-empty credentials work.
                    st.session_state.username = username.strip()
                    st.session_state.logged_in = True
                    st.session_state.page = "predictor"
                    st.rerun()
                else:
                    st.error(
                        "Please enter both username and password."
                    )

        if st.button(
            "← Back to Home",
            key="back_home",
            use_container_width=True
        ):
            st.session_state.page = "home"
            st.rerun()


# =========================================================
# 9. CAR PRICE PREDICTOR
# =========================================================
elif st.session_state.page == "predictor":

    # Protect the predictor from direct navigation
    if not st.session_state.logged_in:
        st.session_state.page = "login"
        st.rerun()

    st.markdown("""
    <div class="hero">
        <h1>Car Price <span class="red">Predictor</span></h1>
        <p>
            Enter your vehicle details to estimate its price.
        </p>
    </div>
    """, unsafe_allow_html=True)

    header_col, logout_col = st.columns([4, 1])

    with header_col:
        st.write(
            f"Welcome, **{st.session_state.username}**!"
        )

    with logout_col:
        if st.button(
            "Logout",
            key="logout",
            use_container_width=True
        ):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.page = "home"
            st.rerun()

    st.divider()

    try:
        car = load_data()
        pipe = load_model()

        required_columns = [
            "name",
            "company",
            "year",
            "kms_driven",
            "fuel_type"
        ]

        if not all(
            column in car.columns
            for column in required_columns
        ):
            st.error(
                "The cleaned dataset must contain: "
                + ", ".join(required_columns)
            )
            st.stop()

        st.markdown(
            "### 🚗 Vehicle Information"
        )

        col1, col2 = st.columns(2)

        with col1:
            companies = sorted(
                car["company"]
                .dropna()
                .astype(str)
                .unique()
            )

            company = st.selectbox(
                "Select Company",
                companies
            )

            models = sorted(
                car.loc[
                    car["company"].astype(str) == company,
                    "name"
                ]
                .dropna()
                .astype(str)
                .unique()
            )

            model = st.selectbox(
                "Select Model",
                models
            )

            valid_years = pd.to_numeric(
                car["year"],
                errors="coerce"
            ).dropna()

            years = sorted(
                valid_years.astype(int).unique(),
                reverse=True
            )

            year = st.selectbox(
                "Year of Purchase",
                years
            )

        with col2:
            fuels = sorted(
                car["fuel_type"]
                .dropna()
                .astype(str)
                .unique()
            )

            fuel = st.selectbox(
                "Fuel Type",
                fuels
            )

            kms = st.number_input(
                "Kilometers Travelled",
                min_value=0,
                max_value=1_000_000,
                value=100,
                step=100
            )

        st.write("")

        if st.button(
            "Predict Price 🚀",
            key="predict_price",
            use_container_width=True
        ):
            input_data = pd.DataFrame(
                [[model, company, year, kms, fuel]],
                columns=[
                    "name",
                    "company",
                    "year",
                    "kms_driven",
                    "fuel_type"
                ]
            )

            predicted_price = float(
                pipe.predict(input_data)[0]
            )

            if predicted_price > 0:
                st.success("Prediction completed!")

                st.metric(
                    label="Estimated Car Price",
                    value=f"₹{predicted_price:,.0f}"
                )

                st.caption(
                    "This is an estimated value from your "
                    "Linear Regression model, not a guaranteed "
                    "market selling price."
                )
            else:
                st.warning(
                    "The model returned a non-positive price. "
                    "Try different car details."
                )

    except FileNotFoundError as error:
        st.error(
            f"Required file not found: {error.filename}. "
            "Place your model and cleaned dataset "
            "in the same folder as app.py."
        )

    except Exception as error:
        st.error(
            f"Prediction could not be completed: {error}"
        )


# =========================================================
# 10. INVALID PAGE FALLBACK
# =========================================================
else:
    st.session_state.page = "welcome"
    st.rerun()
