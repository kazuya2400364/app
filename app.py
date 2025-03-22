import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import pandas as pd
import json
import re
import datetime

def get_gemini_model():
    """Streamlit Cloud の Secrets から Gemini API キーを取得してモデルを初期化"""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel('gemini-1.5-flash')
    except KeyError:
        st.error("Gemini APIキーが Streamlit Cloud の Secrets に設定されていません。")
        return None

def extract_info_with_gemini(model, image_bytes):
    """Gemini API を使用して画像から情報を抽出する関数"""
    if model is None:
        return None
    prompt = """この画像から、以下の情報を抽出して、JSON形式で出力してください。
    抽出する情報:
    - 型番
    - 製造年
    - 定格能力(冷房) (単位も含む)
    - 定格能力(暖房標準) (単位も含む)
    - 定格能力(暖房低温) (単位も含む)
    - 定格消費電力(冷房) (単位も含む)
    - 定格消費電力(暖房標準) (単位も含む)
    - 定格消費電力(暖房低温) (単位も含む)

    出力例:
    {
        "型番": "...",
        "製造年": "...",
        "定格能力(冷房)": "...",
        "定格能力(暖房標準)": "...",
        "定格能力(暖房低温)": "...",
        "定格消費電力(冷房)": "...",
        "定格消費電力(暖房標準)": "...",
        "定格消費電力(暖房低温)": "..."
    }
    """
    response = model.generate_content(
        [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
    )
    response_text = response.text.strip() #前後の空白を削除

    # ```json プレフィックスと ``` サフィックスを取り除く
    response_text = re.sub(r'^```json', '', response_text) #先頭や末尾にある可能性のあるjsonを削除
    response_text = re.sub(r'```$', '', response_text).strip() #前後の空白を削除

    if not response_text:
        st.error("Gemini API からの応答が空です。")
        return None

    try:
        # Gemini の応答が JSON 形式であると期待して解析
        extracted_data = json.loads(response_text)
        return extracted_data
    except Exception as e:
        st.error(f"抽出結果の解析に失敗しました: {e}\n応答内容: {response_text}")
        return None

def find_matching_data(model_list, kw, year): #現在の定格能力と同じ製品を抽出する
    matching_data = model_list[model_list["kW"] == kw]
    if not matching_data.empty:
        if year >= 15:
            matching_data["subsidy"] = matching_data['subsidy_long']
        else:
            matching_data["subsidy"] = matching_data['subsidy_normal']
        return matching_data
    else:
        return "同じ定格能力のエアコンが見つかりませんでした。"

def main():
    st.title("エアコン型番画像から情報を抽出して製品・補助金情報を検索するアプリ")

    # Gemini モデルをセッションステートに保存 (初回のみロード)
    if "gemini_model" not in st.session_state:
        st.session_state["gemini_model"] = get_gemini_model()

    uploaded_file = st.file_uploader("エアコンの型番が写った画像をアップロードしてください", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if st.session_state.get("gemini_model"):
            image = Image.open(uploaded_file)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()

            st.image(image, caption="アップロードされた画像", use_container_width=True)
            st.write("解析中...")

            extracted_info = extract_info_with_gemini(st.session_state["gemini_model"], image_bytes)

            if extracted_info:
                # DataFrame の形式を調整
                df_data = {
                    "型番": [extracted_info.get("型番")],
                    "製造年": [extracted_info.get("製造年")],
                    "定格能力(冷房)": [extracted_info.get("定格能力(冷房)")],
                    "定格能力(暖房標準)": [extracted_info.get("定格能力(暖房標準)")],
                    "定格能力(暖房低温)": [extracted_info.get("定格能力(暖房低温)")],
                    "定格消費電力(冷房)": [extracted_info.get("定格消費電力(冷房)")],
                    "定格消費電力(暖房標準)": [extracted_info.get("定格消費電力(暖房標準)")],
                    "定格消費電力(暖房低温)": [extracted_info.get("定格消費電力(暖房低温)")]
                }
                df = pd.DataFrame(df_data)
                st.subheader("抽出結果 (DataFrame)")
                st.dataframe(df)

                # 定格能力(冷房)と製造年を抽出
                rated_cooling_capacity = extracted_info.get("定格能力(冷房)")
                manufacture_year = extracted_info.get("製造年")

                # 定格能力(冷房) から数値と単位を抽出
                rated_cooling_capacity_num = re.findall(r'\d+\.?\d*',rated_cooling_capacity)

                if rated_cooling_capacity_num:
                    rated_cooling_capacity_kw = float(rated_cooling_capacity_num[0]) # 数値のみ取得
                else:
                    rated_cooling_capacity_kw = 0.0 # 数値がない場合は0.0にする

                # 製造年を数値に変換
                manufacture_year_num = int(re.sub(r"\D", "", manufacture_year)) # 数字以外の文字を削除
                current_year = datetime.date.today().year # 現在の年
                years_passed = current_year - manufacture_year_num

                # 製品・補助金情報を検索
                model_list = pd.read_csv("model_list.csv")
                result = find_matching_data(model_list, rated_cooling_capacity_kw, years_passed)

                st.write("おすすめ製品(現在のエアコンと同じ冷房能力)")
                st.write("補助金申請受付期間は2027年4月30日まで(※上限に到達次第修了)")
                st.dataframe(result[["model", "manufacturer", "kW", "price", "subsidy"]])
                st.link_button("エアコンを注文する", "https://home.tokyo-gas.co.jp/housing/exchange/index.html")
            else:
                st.error("Gemini モデルの初期化に失敗しました。")
        else:
            st.error("Gemini モデルの初期化に失敗しました。")

if __name__ == "__main__":
    main()