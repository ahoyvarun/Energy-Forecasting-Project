# ========== Imports ==========
import pandas as pd
import warnings
from prophet import Prophet
from visualizations import scatter_energy_use, boxplot_energy_intensity, plot_forecast

warnings.filterwarnings("ignore")

# ========== Load & Preview Dataset ==========
df = pd.read_csv("singapore_dataset.csv")
print("Shape:", df.shape)
print(df.columns)
print(df.head())

# ========== Data Cleaning ==========
df_clean = df.dropna(subset=["2017energyuseintensity", "2018energyusintensity"])

# ========== Visualizations ==========
scatter_energy_use(df_clean, "2017energyuseintensity", "2018energyusintensity", "buildingtype")
boxplot_energy_intensity(df_clean)

# ========== Prepare Data for Forecasting ==========
df_grouped = df_clean.groupby("buildingtype")[["2017energyuseintensity", "2018energyusintensity"]].mean().reset_index()
df_melted = df_grouped.melt(id_vars="buildingtype", var_name="year", value_name="eui")
df_melted["year"] = df_melted["year"].str.extract(r"(\d{4})")
df_melted["ds"] = pd.to_datetime(df_melted["year"], format="%Y")
df_melted.rename(columns={"eui": "y"}, inplace=True)
df_melted = df_melted[["buildingtype", "ds", "y"]]
print(df_melted.head())

# ========== Forecast All Building Types ==========
building_types = df_melted["buildingtype"].unique()
all_forecasts = []

for btype in building_types:
    df_bt = df_melted[df_melted["buildingtype"] == btype]
    if len(df_bt) < 2:
        continue
    model = Prophet(yearly_seasonality=False)
    model.fit(df_bt)
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)
    forecast["buildingtype"] = btype
    all_forecasts.append(forecast[["ds", "yhat", "buildingtype"]])
    plot_forecast(model, forecast, title=f"{btype} Energy Use Forecast")

# ========== Save Forecasts to CSV ==========
pd.concat(all_forecasts).to_csv("energy_forecasts_by_type.csv", index=False)