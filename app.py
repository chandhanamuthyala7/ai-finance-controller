from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)


def analyze_finances(df):

    # Clean column names
    df.columns = [col.strip().lower() for col in df.columns]

    required_columns = [
        "date",
        "description",
        "category",
        "type",
        "amount"
    ]

    for col in required_columns:
        if col not in df.columns:
            return None, f"Missing column: {col}"

    # Clean data
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df = df.dropna(subset=["amount"])

    df["type"] = df["type"].str.lower().str.strip()

    # -------------------------
    # Revenue & Expenses
    # -------------------------

    revenue = df[
        df["type"] == "income"
    ]["amount"].sum()

    expenses = df[
        df["type"] == "expense"
    ]["amount"].sum()

    profit = revenue - expenses

    # Profit margin
    if revenue > 0:
        profit_margin = (profit / revenue) * 100
    else:
        profit_margin = 0

    # -------------------------
    # Expense Categories
    # -------------------------

    expense_df = df[
        df["type"] == "expense"
    ].copy()

    category_expenses = (
        expense_df
        .groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    highest_category = "None"

    if len(category_expenses) > 0:
        highest_category = category_expenses.index[0]

    # -------------------------
    # Anomaly Detection
    # -------------------------

    anomalies = []

    if len(expense_df) >= 5:

        model = IsolationForest(
            contamination=0.15,
            random_state=42
        )

        expense_df["anomaly"] = model.fit_predict(
            expense_df[["amount"]]
        )

        anomalies = expense_df[
            expense_df["anomaly"] == -1
        ].to_dict("records")

    # -------------------------
    # AI Recommendations
    # -------------------------

    recommendations = []

    if expenses > revenue:

        recommendations.append(
            "⚠️ Expenses are higher than revenue. "
            "Immediate cost optimization is recommended."
        )

    else:

        recommendations.append(
            "✅ Your business is currently operating "
            "with positive net profit."
        )

    if highest_category != "None":

        highest_amount = category_expenses.iloc[0]

        recommendations.append(
            f"💡 {highest_category} is the highest expense "
            f"category with spending of ₹{highest_amount:,.2f}."
        )

    if len(anomalies) > 0:

        recommendations.append(
            f"🚨 {len(anomalies)} unusual transaction(s) "
            "were detected. Review them carefully."
        )

    else:

        recommendations.append(
            "✅ No major unusual spending patterns detected."
        )

    if profit_margin >= 30:

        recommendations.append(
            "📈 Strong profit margin detected. "
            "Consider investing a portion of the surplus "
            "into business growth."
        )

    elif profit_margin < 15:

        recommendations.append(
            "⚠️ Profit margin is relatively low. "
            "Review recurring and discretionary expenses."
        )

    # -------------------------
    # Chart Data
    # -------------------------

    chart_labels = list(category_expenses.index)
    chart_values = [
        float(value)
        for value in category_expenses.values
    ]

    # -------------------------
    # Recent Transactions
    # -------------------------

    recent_transactions = (
        df.sort_values("date", ascending=False)
        .head(8)
        .to_dict("records")
    )

    return {
        "revenue": float(revenue),
        "expenses": float(expenses),
        "profit": float(profit),
        "profit_margin": float(profit_margin),

        "highest_category": highest_category,

        "category_expenses":
            category_expenses.to_dict(),

        "chart_labels":
            chart_labels,

        "chart_values":
            chart_values,

        "anomalies":
            anomalies,

        "recommendations":
            recommendations,

        "recent_transactions":
            recent_transactions
    }, None


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        if "file" not in request.files:

            error = "Please upload a CSV file."

            return render_template(
                "index.html",
                result=result,
                error=error
            )

        file = request.files["file"]

        if file.filename == "":

            error = "Please select a file."

            return render_template(
                "index.html",
                result=result,
                error=error
            )

        if not file.filename.lower().endswith(".csv"):

            error = "Only CSV files are supported."

            return render_template(
                "index.html",
                result=result,
                error=error
            )

        try:

            df = pd.read_csv(file)

            result, error = analyze_finances(df)

        except Exception as e:

            error = f"Error processing file: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)