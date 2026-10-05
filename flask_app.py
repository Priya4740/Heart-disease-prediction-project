from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import traceback

app = Flask(__name__)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")
ENCODED_FEATURES_PATH = os.path.join(
    BASE_DIR, "encoded_feature_names.pkl"
)
ORIGINAL_FEATURES_PATH = os.path.join(
    BASE_DIR, "original_feature_names.pkl"
)
CATEGORICAL_COLUMNS_PATH = os.path.join(
    BASE_DIR, "categorical_columns.pkl"
)


# =========================================================
# LOAD MODEL FILES
# =========================================================

model = None
encoded_feature_names = []
original_feature_names = []
categorical_columns = []

try:

    model = joblib.load(MODEL_PATH)

    encoded_feature_names = joblib.load(
        ENCODED_FEATURES_PATH
    )

    original_feature_names = joblib.load(
        ORIGINAL_FEATURES_PATH
    )

    categorical_columns = joblib.load(
        CATEGORICAL_COLUMNS_PATH
    )

    print("=" * 60)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 60)
    print("Model classes:", model.classes_)
    print("Original features:", original_feature_names)
    print("Categorical columns:", categorical_columns)
    print("Encoded features:", encoded_feature_names)
    print("=" * 60)

except Exception as e:

    print("=" * 60)
    print("MODEL LOADING ERROR")
    print("=" * 60)
    print(str(e))
    traceback.print_exc()
    print("=" * 60)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        features=original_feature_names
    )


# =========================================================
# PREDICT
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------------
        # CHECK MODEL
        # -------------------------------------------------

        if model is None:

            raise Exception(
                "Model could not be loaded. "
                "Check your .pkl files."
            )


        # -------------------------------------------------
        # COLLECT INPUT
        # -------------------------------------------------

        input_data = {}

        for feature in original_feature_names:

            value = request.form.get(feature)

            if value is None:

                raise Exception(
                    "Missing input field: " + feature
                )

            value = value.strip()

            if value == "":

                raise Exception(
                    "Empty input field: " + feature
                )

            input_data[feature] = value


        # -------------------------------------------------
        # FIND NUMERIC FEATURES
        # -------------------------------------------------

        numeric_columns = []

        for feature in original_feature_names:

            if feature not in categorical_columns:

                numeric_columns.append(feature)


        # -------------------------------------------------
        # CONVERT NUMERIC VALUES
        # -------------------------------------------------

        for column in numeric_columns:

            try:

                input_data[column] = float(
                    input_data[column]
                )

            except Exception:

                raise Exception(
                    "Invalid numeric value for: "
                    + column
                )


        # -------------------------------------------------
        # DATAFRAME
        # -------------------------------------------------

        input_df = pd.DataFrame(
            [input_data],
            columns=original_feature_names
        )


        print("=" * 60)
        print("INPUT DATA")
        print("=" * 60)
        print(input_df)
        print("=" * 60)


        # -------------------------------------------------
        # ENCODE CATEGORICAL DATA
        # -------------------------------------------------

        input_encoded = pd.get_dummies(
            input_df,
            columns=categorical_columns,
            drop_first=False
        )


        # -------------------------------------------------
        # MATCH MODEL FEATURES
        # -------------------------------------------------

        input_encoded = input_encoded.reindex(
            columns=encoded_feature_names,
            fill_value=0
        )


        # -------------------------------------------------
        # CONVERT TO FLOAT
        # -------------------------------------------------

        input_encoded = input_encoded.astype(float)


        print("Encoded shape:")
        print(input_encoded.shape)

        print("Expected features:")
        print(len(encoded_feature_names))


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        prediction = model.predict(
            input_encoded
        )[0]


        probabilities = model.predict_proba(
            input_encoded
        )[0]


        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        probability_dict = dict(
            zip(
                model.classes_,
                probabilities
            )
        )


        class_0_probability = (
            probability_dict.get(0, 0) * 100
        )

        class_1_probability = (
            probability_dict.get(1, 0) * 100
        )


        # -------------------------------------------------
        # CLASS INTERPRETATION
        #
        # Current application mapping:
        #
        # 0 = HIGHER RISK
        # 1 = LOWER RISK
        #
        # -------------------------------------------------

        if int(prediction) == 0:

            result = "Higher-Risk Class"
            result_class = "high"

            displayed_probability = (
                class_0_probability
            )

        elif int(prediction) == 1:

            result = "Lower-Risk Class"
            result_class = "low"

            displayed_probability = (
                class_1_probability
            )

        else:

            result = "Unknown Class"
            result_class = "error"

            displayed_probability = 0


        # -------------------------------------------------
        # PRINT RESULT
        # -------------------------------------------------

        print("=" * 60)
        print("PREDICTION")
        print("=" * 60)

        print("Prediction:", prediction)

        print(
            "Class 0:",
            round(class_0_probability, 2),
            "%"
        )

        print(
            "Class 1:",
            round(class_1_probability, 2),
            "%"
        )

        print("Result:", result)

        print("=" * 60)


        # -------------------------------------------------
        # RESULT PAGE
        # -------------------------------------------------

        return render_template(
            "result.html",

            result=result,

            result_class=result_class,

            probability=round(
                displayed_probability,
                2
            ),

            class_0_probability=round(
                class_0_probability,
                2
            ),

            class_1_probability=round(
                class_1_probability,
                2
            ),

            error=None
        )


    except Exception as e:

        # -------------------------------------------------
        # PRINT COMPLETE ERROR
        # -------------------------------------------------

        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        print(str(e))

        traceback.print_exc()

        print("=" * 60)


        # -------------------------------------------------
        # SHOW ERROR PAGE
        # -------------------------------------------------

        return render_template(

            "result.html",

            result="Prediction Error",

            result_class="error",

            probability=0,

            class_0_probability=0,

            class_1_probability=0,

            error=str(e)

        )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )