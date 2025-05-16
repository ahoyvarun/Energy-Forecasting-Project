import matplotlib.pyplot as plt
import seaborn as sns

def scatter_energy_use(df, x_col, y_col, hue_col, diagonal_max=3000):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
    plt.plot([0, diagonal_max], [0, diagonal_max], ls="--", c="gray")
    plt.title("Building Energy Use Intensity: 2017 vs 2018")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def boxplot_energy_intensity(df, buildingtype_col="buildingtype", intensity_cols=["2017energyuseintensity", "2018energyusintensity"]):
    melted_df = df.melt(id_vars=[buildingtype_col], value_vars=intensity_cols, var_name="year", value_name="energy_intensity")
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=melted_df, x=buildingtype_col, y="energy_intensity", hue="year")
    plt.title("Energy Use Intensity Distribution by Building Type (2017 vs 2018)")
    plt.ylabel("Energy Use Intensity (kWh/m²/year)")
    plt.xlabel("Building Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_forecast(model, forecast, title="Forecast Plot"):
    fig = model.plot(forecast)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("EUI (kWh/m²/year)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()