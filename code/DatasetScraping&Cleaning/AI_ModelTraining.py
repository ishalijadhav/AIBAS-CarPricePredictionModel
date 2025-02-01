import keras
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import statsmodels.api as sm

train_data = pd.read_csv('../../data/training_data.csv')
test_data = pd.read_csv('../../data/test_data.csv')

# Separating independent and dependent variable
x_train = train_data.drop('selling_price', axis=1)  # Independent variables
y_train = train_data['selling_price']  # Dependent variable

x_test = test_data.drop('selling_price', axis=1)
y_test = test_data['selling_price']

def build_model():

    model = keras.Sequential([
    Dense(64, input_dim=x_train.shape[1], activation='relu'),  # Input layer
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
    ])

    model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
    )

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
    )

    model.save('currentAiSolution.keras')
    return model, history


def plot_training_curves(epochs, loss, val_loss, metric, val_metric):

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training')
    plt.plot(epochs, val_loss, label='Validation')
    plt.title(f'Loss (Epochs: {len(epochs)})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # Metric plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metric, label='Training')
    plt.plot(epochs, val_metric, label='Validation')
    plt.title(f'MAE (Epochs: {len(epochs)})')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.savefig('../../documentation/training_curves.png',dpi=300)
    plt.close()

def evaluate_model(model, history):
    # Get actual epochs trained
    actual_epochs = len(history.history['loss'])

    # Generate visualizations
    plot_training_curves(
        epochs=range(1, actual_epochs + 1),
        loss=history.history['loss'],
        val_loss=history.history['val_loss'],
        metric=history.history['mean_absolute_error'],  # MAE in this case
        val_metric=history.history['val_mean_absolute_error']
    )


def save_training_report(history, path='../../documentation/'):
    report = {
        'total_epochs': len(history.history['loss']),
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_train_mae': history.history['mean_absolute_error'][-1],
        'final_val_mae': history.history['val_mean_absolute_error'][-1]
    }

    pd.DataFrame(report, index=[0]).to_csv(f'{path}/training_report.csv', index=False)


def plot_diagnostic_plots():
    # Load model and make predictions
    model = load_model("currentAiSolution.keras")
    y_pred = model.predict(x_test).flatten()
    residuals = y_test - y_pred
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    # Calculate leverage using hat matrix
    X = sm.add_constant(np.column_stack([np.ones_like(y_pred), y_pred]))  # Simple example matrix
    hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diag(hat_matrix)

    #Cook's distance
    cooks_d = (standardized_residuals ** 2) * leverage / (X.shape[1] * (1 - leverage) ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residual vs Fitted values
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residual vs Fitted Values')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')

    # 2. Square Root of Standardized Residuals vs Fitted values
    sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))
    axes[0, 1].scatter(y_pred, sqrt_abs_resid, alpha=0.5)
    axes[0, 1].set_title('Scale-Location (Sqrt Standardized Residuals vs Fitted)')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('√|Standardized Residuals|')

    # 3. Q-Q plot of Standardized Residuals
    sm.qqplot(standardized_residuals, line='45', fit=True, alpha=0.5, ax=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q Plot')
    axes[1, 0].set_xlabel('Theoretical Quantiles')
    axes[1, 0].set_ylabel('Standardized Residuals')

    # 4. Residual vs Leverage with Cook's distance contours
    scatter = axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.5, c=cooks_d, cmap='viridis')
    axes[1, 1].set_title('Residual vs Leverage')
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Standardized Residuals')

    # Add Cook's distance contours
    x = np.linspace(min(leverage), max(leverage), 50)
    for c in [0.5, 1]:
        axes[1, 1].plot(x, np.sqrt((c * X.shape[1] * (1 - x) ** 2) / x),
                        label=f"Cook's D={c}", linestyle='--', color='red')
        axes[1, 1].plot(x, -np.sqrt((c * X.shape[1] * (1 - x) ** 2) / x),
                        linestyle='--', color='red')

    axes[1, 1].legend()
    fig.colorbar(scatter, ax=axes[1, 1], label="Cook's Distance")

    plt.tight_layout()

    plt.savefig('../../documentation/diagnostic_plots.png', dpi=300)

def plot_scatter_with_regression():

    # Load model and make predictions
    model = load_model("currentAiSolution.keras")
    y_pred = model.predict(x_test).flatten()

    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ci=None)

    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('Scatter Plot of y_test vs y_pred with Regression Line')
    plt.grid(True)

    # Save plot
    plt.savefig('../../documentation/scatter_regression_plot.png', dpi=300)


# Call function to generate and save the plot


if __name__ == "__main__":
    # Train model and get history
    trained_model, training_history = build_model()

    # Save epoch-related data
    save_training_report(training_history)

    #Diagonostic plot
    plot_diagnostic_plots()

    #Scatter plot
    plot_scatter_with_regression()

    # Evaluate with epoch context
    evaluate_model(trained_model, training_history)








