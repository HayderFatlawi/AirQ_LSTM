import os
import requests
import joblib
import numpy as np
import json
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for
from tensorflow.keras.models import load_model

# محاولة استيراد دالة التدريب
try:
    from main import train_city_model
except ImportError:
    # في حال حذف ملف main.py، نعيد رسالة خطأ عند محاولة طلب التدريب
    def train_city_model(city, start_date): 
        yield json.dumps({"error": "ملف التدريب main.py غير موجود! لا يمكن بدء تدريب جديد."})

app = Flask(__name__)

# خازن للموديلات لتقليل وقت التحميل
assets_cache = {}

def get_city_assets(city_name):
    """تحميل الموديل والمقاييس من فولدر المحافظة"""
    if city_name in assets_cache:
        return assets_cache[city_name]

    folder = city_name
    model_path = os.path.join(folder, "air_quality_model.keras")
    sx_path = os.path.join(folder, "scaler_X.pkl")
    sy_path = os.path.join(folder, "scaler_y.pkl")

    # نتحقق من وجود الملفات الثلاثة الأساسية للتنبؤ
    if all(os.path.exists(p) for p in [model_path, sx_path, sy_path]):
        try:
            assets = {
                "model": load_model(model_path, compile=False),
                "scaler_x": joblib.load(sx_path),
                "scaler_y": joblib.load(sy_path)
            }
            assets_cache[city_name] = assets
            return assets
        except Exception as e:
            print(f"⚠️ خطأ تحميل {city_name}: {e}")
    return None

# =========================================================
# 2. نظام النصائح (Expert System)
# =========================================================
def get_ai_advice(aqi, city):
    if aqi <= 50:    state, color = "good", "🟢"
    elif aqi <= 100: state, color = "moderate", "🟡"
    elif aqi <= 150: state, color = "unhealthy_sensitive", "🟠"
    elif aqi <= 200: state, color = "unhealthy", "🔴"
    else:            state, color = "hazardous", "🟣"

    advice_map = {
        "good": {
            "allergy": "الجو يجنن وصافي بـ {city}، ماراح يضوجكم صدركم اليوم.",
            "normal": "الهواء نقي ويخبل، استغلوه للسفرات وافتحوا شبابيك البيت."
        },
        "moderate": {
            "allergy": "الجو بي شوية تراب ناعم بـ {city}، خلو الكمامة قريبة احتياط.",
            "normal": "الجو مقبول، بس حاولوا ما تجهدون نفسكم بالرياضة برا هواية."
        },
        "unhealthy_sensitive": {
            "allergy": "الجو يغث ويخنگ شوية بـ {city}، أهل الحساسية ابقوا بمكان بي تبريد.",
            "normal": "التلوث صاعد، إذا تطلعون وقت طويل البسوا كمامة لأن الهوى ثقيل."
        },
        "unhealthy": {
            "allergy": "تحذير! الجو مو تمام بـ {city}، الطلعة بيها خطر على صدركم.",
            "normal": "التلوث عالي ويسبب ضيق تنفس، تجنبوا الشغل التعب برا وصحتكم أهم."
        },
        "hazardous": {
            "allergy": "وضع خطر! الجو مسموم بـ {city}، لا تطلعون من البيت أبد وسدوا الشبابيك.",
            "normal": "كارثة بيئية! الهواء خطر جداً على الكل، الطلعة ممنوعة إلا للضرورة القصوى."
        }
    }
    return f"لأهل الحساسية: {advice_map[state]['allergy'].format(city=city)} | نصيحة عامة: {advice_map[state]['normal'].format(city=city)}"

# =========================================================
# 3. وظائف جلب البيانات والتنبؤ
# =========================================================
def fetch_location_data(lat, lon):
    try:
        aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&past_days=1&forecast_days=2"
        aq_res = requests.get(aq_url).json()

        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&current_weather=true&past_days=1&forecast_days=2"
        w_res = requests.get(w_url).json()
        
        return aq_res.get('hourly'), w_res.get('hourly'), w_res.get('current_weather')
    except Exception as e:
        print(f"❌ خطأ API: {e}")
        return None, None, None

def run_prediction(city, aq, w, idx):
    assets = get_city_assets(city)
    if not assets: return 0.0
    try:
        sequence = []
        for i in range(idx - 24, idx):
            row = [
                aq['carbon_monoxide'][i], aq['nitrogen_dioxide'][i], 
                aq['sulphur_dioxide'][i], aq['ozone'][i], aq['pm2_5'][i], 
                aq['pm10'][i], w['temperature_2m'][i], 
                w['relative_humidity_2m'][i], w['wind_speed_10m'][i]
            ]
            sequence.append(row)
        
        scaled_data = assets['scaler_x'].transform(sequence)
        prediction = assets['model'].predict(scaled_data.reshape(1, 24, 9), verbose=0)
        final_val = assets['scaler_y'].inverse_transform(prediction)[0][0]
        return round(max(0, float(final_val)), 2)
    except:
        return 0.0

# =========================================================
# 4. مسارات التطبيق (Routes)
# =========================================================

@app.route("/")
def home():
    """التحقق من جاهزية كافة الموديلات قبل التوجيه"""
    cities = ["Baghdad", "Basra", "Najaf"]
    for city in cities:
        # إذا نقص ملف الموديل في أي محافظة، يتم التوجيه لصفحة التدريب
        if not os.path.exists(os.path.join(city, "air_quality_model.keras")):
            return redirect(url_for('train_page'))
    return redirect(url_for('prediction_page'))

@app.route("/train_page")
def train_page():
    return render_template("train.html")

@app.route("/prediction_page")
def prediction_page():
    return render_template("prediction.html")

@app.route("/get_aqi_forecast")
def get_aqi_forecast():
    city = request.args.get('city', "Baghdad")
    coords_map = {"Baghdad": (33.34, 44.40), "Basra": (30.50, 47.81), "Najaf": (32.02, 44.33)}
    lat, lon = coords_map.get(city, (33.34, 44.40))

    aq_h, w_h, cur_w = fetch_location_data(lat, lon)
    if not aq_h or not w_h:
        return jsonify({"error": True, "message": "فشل جلب البيانات"})

    current_idx = 24 
    short_forecast = [run_prediction(city, aq_h, w_h, current_idx + i) for i in range(-6, 7)]
    short_labels = [f"{i}h" if i != 0 else "الآن" for i in range(-6, 7)]
    long_forecast = [run_prediction(city, aq_h, w_h, current_idx + i) for i in range(1, 13)]
    long_labels = [f"+{i}h" for i in range(1, 13)]

     # تعديل بسيط داخل مسار get_aqi_forecast في بايثون
    pollutants = {
        "PM2.5": round(aq_h['pm2_5'][current_idx], 2),
        "PM10":  round(aq_h['pm10'][current_idx], 2),
        "CO":    round(aq_h['carbon_monoxide'][current_idx], 2),
        "NO2":   round(aq_h['nitrogen_dioxide'][current_idx], 2)
    }

    advice = get_ai_advice(short_forecast[6], city)

    return jsonify({
        "status": "success",
        "city": city,
        "current_weather": {
            "temp": cur_w['temperature'],
            "wind": cur_w['windspeed'],
            "humidity": w_h['relative_humidity_2m'][current_idx], 
            "pollutants": pollutants
        },
        "short_term": { "labels": short_labels, "values": short_forecast },
        "long_term": { "labels": long_labels, "values": long_forecast },
        "ai_advice": advice
    })

@app.route("/train")
def train():
    city = request.args.get('city', 'Baghdad')
    start_date = request.args.get('start_date', '2015-01-01')
    return Response(train_city_model(city, start_date), mimetype="application/json")

@app.route("/check_models")
def check_models():
    """هذا المسار هو المسؤول عن تحديث البطاقات (Cards) في الواجهة"""
    cities = ["Baghdad", "Basra", "Najaf"]
    status = {}
    all_ready = True
    for city in cities:
        # نتحقق من وجود ملف الموديل الفعلي داخل مجلد كل مدينة
        model_exists = os.path.exists(os.path.join(city, "air_quality_model.keras"))
        status[city] = model_exists
        if not model_exists: 
            all_ready = False
            
    # نرسل الحالة لكل مدينة (True يعني Model Ready، False يعني Needs Training)
    return jsonify({"models": status, "all_ready": all_ready})

if __name__ == "__main__":
    app.run(debug=True, port=5000)