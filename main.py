import time, os, json, joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def build_model(time_step, feature_count):
    model = Sequential([
        Input(shape=(time_step, feature_count)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")
    return model

def train_city_model(city_name, start_date):
    # Ensure directory exists for each city
    os.makedirs(city_name, exist_ok=True)
    file_path = os.path.join(city_name, "data.csv")

    if not os.path.exists(file_path):
        yield json.dumps({"error": f"Data for {city_name} not found!"}) + "\n"
        return

    # Data loading and filtering by date
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[data['Date'] >= start_date].ffill().bfill()

    # 9 Features logic
    features = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10", "Temp", "Humidity", "Wind"]
    X = data[features].values
    y = data["AQI"].values.reshape(-1, 1)

    sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
    X_scaled, y_scaled = sc_X.fit_transform(X), sc_y.fit_transform(y)
    
    time_step = 24
    Xs, ys = [], []
    for i in range(len(X_scaled) - time_step):
        Xs.append(X_scaled[i : i + time_step])
        ys.append(y_scaled[i + time_step])

    X_train, X_test, y_train, y_test = train_test_split(np.array(Xs), np.array(ys), test_size=0.15, shuffle=False)
    model = build_model(time_step, X_train.shape[2])
    
    best_r2, wait, patience = -np.inf, 0, 10

    for epoch in range(100):
        start_time = time.time()
        model.fit(X_train, y_train, epochs=1, batch_size=256, verbose=0)

        pred = sc_y.inverse_transform(model.predict(X_test, verbose=0)).flatten()
        actual = sc_y.inverse_transform(y_test).flatten()

        r2 = float(r2_score(actual, pred))
        r2_display = r2 if r2 > 0 else 0.01
        epoch_duration = round(time.time() - start_time, 2)

        # منطق الحفظ وعداد الصبر الذكي
        if r2 > best_r2:
            best_r2 = r2
            joblib.dump(sc_X, os.path.join(city_name, "scaler_X.pkl"))
            joblib.dump(sc_y, os.path.join(city_name, "scaler_y.pkl"))
            model.save(os.path.join(city_name, "air_quality_model.keras"))
            
        else:
            # إذا لم تتحسن الدقة وكانت الدقة الحالية فوق الـ 90%، نبدأ نعد صبر
            if r2 >= 0.90:
                wait += 1
            else:
                wait = 0 # إذا نزلت تحت الـ 90% نصفر العداد

        status_data = {
            "city": city_name, 
            "epoch": epoch + 1,
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "mae": float(mean_absolute_error(actual, pred)),
            "r2": r2_display,
            "wait": wait, 
            "epoch_time": epoch_duration,
            "actual": actual[:50].tolist(), 
            "predicted": pred[:50].tolist()
        }

        # شروط التوقف النهائي
        if r2 >= 0.96:
            status_data.update({"status": "stabilized", "message": "تم الوصول للهدف (96%)"})
            yield json.dumps(status_data) + "\n"
            break
            
        if wait >= patience:
            status_data.update({"status": "stabilized", "message": "استقرار 10 دورات فوق 90%"})
            yield json.dumps(status_data) + "\n"
            break

        yield json.dumps(status_data) + "\n"