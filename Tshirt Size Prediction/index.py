# Machine learning model to predict if the size is small, medium or large

# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from nicegui import ui

# White-background theme colors
THEME_COLORS = {
    "primary": "#2563EB",
    "primary_dark": "#1D4ED8",
    "primary_light": "#DBEAFE",
    "border": "#BFDBFE",
    "text": "#1E293B",
    "muted_text": "#64748B",
    "surface": "#FFFFFF",
    "page_background": "#F8FAFC",
}

# Load training data
df = pd.read_csv("tshirt_sizes.csv")

# Making a Header
ui.dark_mode(False)
ui.query("body").style(f"background-color: {THEME_COLORS['page_background']};")

with ui.header().classes("text-white p-4 items-center justify-between").style(
    f"background-color: {THEME_COLORS['primary']}; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.18);"
):
    with ui.row().classes("items-center gap-3"):
        ui.icon("bolt", size="md").classes("text-white-400")
        ui.label("Tshirt Size Predictor").classes("text-xl font-bold tracking-wide")


# Preprocessing the data
le = LabelEncoder()
encoded_sizes = le.fit_transform(df['size'])

# Prepare features and target
X = df[['height_cm', 'weight_kg']]
y = encoded_sizes

# Training-Testing the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=250)
model.fit(X_train, y_train)

# Test metrics and classification report
predictions = model.predict(X_test)
test_r2 = model.score(X_test, y_test)
test_accuracy = accuracy_score(y_test, predictions)
print(f"R2 Score: {test_r2}")
print(f"Accuracy: {test_accuracy}")
print(f"Classification Report: \n {classification_report(y_test, predictions)}")

# UI of the Predictor
with ui.column().classes("w-full gap-4 p-4"):
    with ui.grid(columns="repeat(5, minmax(0, 1fr))").classes("w-full gap-4"):
        card_style = (
            f"background-color: {THEME_COLORS['surface']}; "
            f"border: 1px solid {THEME_COLORS['border']}; "
            "box-shadow: 0 4px 14px rgba(30, 41, 59, 0.08);"
        )

        with ui.card().classes("w-full rounded-xl p-6 col-span-2").style(card_style):
            ui.label("Height").classes("text-lg font-bold mb-4").style(f"color: {THEME_COLORS['text']}")
            height_input = ui.input(
                "Height (cm)",
                placeholder="Enter height in cm",
            ).props("outlined dense type=number")
            height_input.classes("w-full")
            with ui.card().classes("w-full items-center p-3 mt-4").style(
                f"background-color: {THEME_COLORS['primary_light']}; border: none;"
            ):
                ui.label("R2 Score").classes("text-sm font-bold").style(
                    f"color: {THEME_COLORS['muted_text']}"
                )
                ui.label(f"{test_r2:.2%}").classes("text-xl font-bold").style(
                    f"color: {THEME_COLORS['primary_dark']}"
                )

        with ui.card().classes("w-full rounded-xl p-6 col-span-2").style(card_style):
            ui.label("Weight").classes("text-lg font-bold mb-4").style(f"color: {THEME_COLORS['text']}")
            weight_input = ui.input(
                "Weight (kg)",
                placeholder="Enter weight in kg",
            ).props("outlined dense type=number")
            weight_input.classes("w-full")

        with ui.card().classes("w-full rounded-xl p-6 col-span-1 items-center").style(card_style):
            ui.label("Accuracy").classes("text-lg font-bold mb-4").style(f"color: {THEME_COLORS['text']}")
            ui.label(f"{test_accuracy:.2%}").classes("text-xl font-bold").style(
                f"color: {THEME_COLORS['primary_dark']}"
            )

        with ui.card().classes("w-full rounded-xl p-6 col-span-5 items-center").style(card_style):
            ui.label("Predicted Size").classes("text-lg font-bold").style(f"color: {THEME_COLORS['text']}")
            prediction_label = ui.label("—").classes("text-3xl font-bold mt-2").style(
                f"color: {THEME_COLORS['primary']}"
            )

    error_label = ui.label().classes("text-red-600 font-bold")

    def predict_size():
        try:
            height = float(height_input.value)
            weight = float(weight_input.value)
            if height <= 0 or weight <= 0:
                raise ValueError

            new_tshirt_data = pd.DataFrame(
                [[height, weight]], columns=['height_cm', 'weight_kg']
            )
            prediction = model.predict(new_tshirt_data)[0]
            predicted_size = le.inverse_transform([prediction])[0]
            
            if (predicted_size == "S"):
                predicted_size_label = "Small"
            elif (predicted_size == "M"):
                predicted_size_label = "Medium"
            elif (predicted_size == "L"):
                predicted_size_label = "Large"
            else:
                pass
            
            prediction_label.set_text(predicted_size_label)
            error_label.set_text("")

            new_row = pd.DataFrame([{
                'height_cm': height,
                'weight_kg': weight,
                'size': predicted_size,
            }])
            new_row.to_csv(
                "tshirt_sizes.csv", mode="a", header=False,
                index=False, lineterminator='\n'
            )
            print(f"The tshirt size is {predicted_size}.")
        except (TypeError, ValueError):
            prediction_label.set_text("—")
            error_label.set_text("Please enter a valid height and weight.")

    ui.button("Predict T-shirt Size", on_click=predict_size).classes("w-full mt-2 py-3 rounded-lg").style(
        f"background-color: {THEME_COLORS['primary_dark']}; color: white; font-weight: 700;"
    )


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Tshirt Size Predictor", port=8080, reload=True)