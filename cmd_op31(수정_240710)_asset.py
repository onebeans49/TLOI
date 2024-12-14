import pymysql
import numpy as np
import pandas as pd
import joblib
import shap
import schedule
import time
import dice_ml
import warnings
import logging
import random

warnings.filterwarnings("ignore")
np.random.seed(42)


# 로그 설정
logging.basicConfig(
    filename='OP30-1to40_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# 데이터베이스 연결 정보
db_config = {
    "host": "host",
    "user": "user",
    "password": "password",
    "database": "database",
    "port": 'port'
}

# 마지막으로 가져온 데이터 저장 변수
last_data = None

def fetch_data(table_name):
    try:
        # 데이터베이스 연결
        conn = pymysql.connect(**db_config)

        with conn.cursor() as cursor:
            # SQL 쿼리 실행
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            rows = cursor.fetchall()

            # 컬럼 이름 가져오기
            column_names = [desc[0] for desc in cursor.description]

        # 데이터프레임으로 변환
        df = pd.DataFrame(rows, columns=column_names)
        return df

    except Exception as e:
        print(f"Error fetching data from {table_name}:", e)
        logging.error(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()

    finally:
        conn.close()

def custom(x):
    if x < 0:
        return np.floor(x * 100) / 100  # 음수는 내림 처리
    else:
        return np.ceil(x * 100) / 100  # 양수는 올림 처리

def process_new_data():
    global last_data

    try:
        # 데이터 가져오기
        df_op21 = fetch_data('tbl_op20_1')
        df_op31 = fetch_data('tbl_op30_1')

        if df_op31.empty:
            return

        current_no = df_op31['no'].max()
        print(f"데이터 업데이트 확인중.. 현재 no: {current_no}..")

        if last_data is None:
            last_data = current_no - 1

        if current_no != last_data:
            
            if last_data + 1 == current_no:
                last_data = current_no
            else:
                last_data = last_data + 1
            
            print(f"새로운 데이터 확인.. 현재 no: {last_data}..")
        
        else:
            return

        # current_no = df_op31['no'].max()
        # print(f"데이터 업데이트 확인중.. 현재 no: {current_no}..")

        # if last_data is None or current_no != last_data:
        #     print(f"새로운 데이터 확인.. 현재 no: {current_no}..")
        #     last_data = current_no

        ''' 가져온 데이터 전처리 '''
        df_op21 = df_op21[['D5', 'D12','D15','D21','D33']]
        df_op31 = df_op31[['D5', 'D12','D15','D21','D33']]

        df_op21.rename(columns={"D5": "튜브바코드", "D12": "OP20-1전장","D15": "OP20-1거리","D21": "OP20-1좌1차피크","D33": "OP20-1우1차피크"}, inplace=True)
        df_op31.rename(columns={"D5": "튜브바코드", "D12": "OP30-1전장","D15": "OP30-1거리","D21": "OP30-1좌1차피크","D33": "OP30-1우1차피크"}, inplace=True)

        # OP Data Integration
        df_op2131t = pd.merge(left=df_op21, right=df_op31, how="right", on="튜브바코드")
        df_op2131t = df_op2131t.fillna(0)

        x = df_op2131t[["OP20-1전장", "OP20-1거리",  "OP20-1좌1차피크", "OP20-1우1차피크", "OP30-1전장", "OP30-1거리", "OP30-1좌1차피크", "OP30-1우1차피크"]]


        ''' 학습된 모델 불러온 후 예측하기 '''
        # 모델 불러오기
        loaded_model = joblib.load('model31.joblib')

        # 불러온 모델로 예측 확률 계산
        loaded_proba = loaded_model.predict_proba(x)
        loaded_pred = loaded_model.predict(x)

        ''' 예측한 결과 기반으로 수치 산출 '''
        # 예측할 Index
        index = last_data-1

        # 합격 예상 확률 추출
        class_1_probability = np.round(loaded_proba[:, 1], 4)
        prob = class_1_probability[index]
        print()
        print('예상 합격률:', prob)

        # SHAP 값 계산을 위한 Explainer 생성
        explainer = shap.TreeExplainer(loaded_model)

        # 특정 샘플에 대한 SHAP 값을 계산
        shap_values = explainer.shap_values(x.iloc[[index]], approximate=True)

        # 합격에 대한 SHAP 값 선택
        shap_values_sample = shap_values[1][0]

        # SHAP 값을 데이터프레임으로 변환
        shap_df = pd.DataFrame({
            'Feature': x.columns,
            'SHAP Value': shap_values_sample,
        })

        # SHAP이 낮은 칼럼(불량 판정에 큰 영향을 미친 요인)
        low_shap = shap_df[shap_df['SHAP Value'] <= -0.1]

        if low_shap.empty:  # -0.1 이하가 없으면 최솟값만 도출
            low_shap = shap_df.loc[[shap_df['SHAP Value'].idxmin()]]

        x_low = x[low_shap['Feature']].values  # low_shap의 원본 수치 추적

        prob_item = low_shap['Feature'].tolist()  # low_shap의 칼럼명
        prob_value = x_low[index].tolist()  # 수치

        # 결과문장 다듬기

        prob_item_str = ', '.join(prob_item)
        prob_value_str = ', '.join(map(str, prob_value))

        # 반사실적 설명(Counterfactual Explantation)을 위한 예측 값을 담은 df 생성
        predicted_df = x.copy()
        predicted_df['predicted'] = loaded_pred

        # 예측 확률이 0.9 미만인 경우
        if prob < 0.9:

            #Counterfactual Explanations 생성
            data_dice = dice_ml.Data(dataframe=predicted_df,
                                     continuous_features=["OP20-1전장", "OP20-1거리", "OP20-1좌1차피크", "OP20-1우1차피크","OP30-1전장", "OP30-1거리", "OP30-1좌1차피크", "OP30-1우1차피크"],
                                     outcome_name='predicted')
            model_dice = dice_ml.Model(model=loaded_model, backend="sklearn")
            explainer_dice = dice_ml.Dice(data_dice, model_dice, method='random')

            # 각 피처에 대해 상한과 하한을 설정
            feature_ranges = {
                'OP20-1전장': [-125, 125],
                'OP20-1거리': [-0.15, 0.15],
                'OP20-1좌1차피크': [150, 600],
                'OP20-1우1차피크': [150, 600],
                'OP30-1전장': [120, 128],
                'OP30-1거리': [-0.15, 0.15],
                'OP30-1좌1차피크': [150, 600],
                'OP30-1우1차피크': [150, 600],
            }

            # 특정 피처만 변경 가능하도록 permitted_range 설정
            permitted_range = {}
            for feature in feature_ranges:
                if feature in prob_item:
                    permitted_range[feature] = feature_ranges[feature]  # 불량 인자 피처만 변경 가능하도록 설정
                else:
                    permitted_range[feature] = [x.loc[index, feature], x.loc[index, feature]]  # 나머지 피처는 고정

            # Counterfactual Explanations 계산
            cf = explainer_dice.generate_counterfactuals(x.loc[[index]], total_CFs=3, desired_class=1,
                                                             permitted_range=permitted_range)

            # Counterfactual Explanations 데이터프레임으로 저장
            cf_df_variable = cf.cf_examples_list[0].final_cfs_df

            # 각 계산 값에 유클리드 거리 계산
            distances = cf_df_variable[prob_item].apply(
                lambda row: np.linalg.norm(row - x.iloc[index][prob_item].values), axis=1)

            # 0보다 큰 거리들만 필터링 후 가장 작은 값으로 솔루션 선택
            positive_distances = distances[distances > 0]

            if not positive_distances.empty:
                closest_cf = cf_df_variable.loc[positive_distances.idxmin()]
                diffs = closest_cf[prob_item] - x.iloc[index][prob_item]

                # 하나의 솔루션 생성
                solution_feature = prob_item[np.argmin(diffs)]  # 가장 큰 차이를 보이는 피처 선택
                solution_diff = diffs[np.argmin(diffs)]  # 해당 피처의 차이 값

                if solution_diff == 0 and all(v == 0 for v in prob_value):
                    solution_str = '데이터가 누락되어 결과를 제공할 수 없습니다.'
                elif solution_diff == 0:
                    diffs = diffs[diffs != 0]
                    solution_diff = diffs[np.argmin(diffs)]
                    solution_diff = custom(solution_diff)
                    solution_str = f"'{solution_feature}'를 {solution_diff:.2f}만큼 수정하세요."
                else:
                    solution_diff = custom(solution_diff)
                    solution_str = f"'{solution_feature}'를 {solution_diff:.2f}만큼 수정하세요."

                for item, value in zip(prob_item, prob_value):
                    prob_item_str = item
                    prob_value_str = str(value)
                    item_solution = solution_str if item == solution_feature else ""  # 솔루션과 불량 인자 매칭, 매칭되지 않은 항목은 공백

        ''' 결과 값 도출 '''
        print()
        if prob >= 0.9:
            print(f"합격률은 {prob * 100}%로 예측됩니다.")
        else:
            print(f"합격률은 {prob * 100}%로 예측되며 불량인자는 {prob_item_str}이(가) {prob_value_str}이기 때문으로 예측됩니다.")
        print('-----------------------')
        print('현재 index', index)        

        ''' DB에 다시 접속 후 UPDATE 및 INSERT '''
        try:
            # 데이터베이스 연결
            conn = pymysql.connect(**db_config)
            with conn.cursor() as cursor:
                # UPDATE SQL 쿼리 실행
                query1 = "UPDATE tbl_op30_1 SET passing_rate = %s WHERE no = %s;"
                query2 = "INSERT INTO tbl_op30_1_defective_item (op30_1_no, item_name, item_value, item_solution) VALUES (%s, %s, %s, %s)"
                query3 = 'INSERT INTO tbl_op30_1_asset_defective_item (op30_1_no, item_name) VALUES (%s, %s);'

                cursor.execute(query1, (prob * 100, index + 1))
                
                print('-----------------------')
                print('현재 index', index)

                if prob < 0.9:
                    for i, (item, value, solution) in enumerate(zip(prob_item, prob_value, item_solution)):
                        cursor.execute(query2, (index + 1, item, value, item_solution))

                print('-----------------------')
                print('현재 index', index)

                if prob <= 0.5:
                    asset = ['제어', '센서', '실린더', '밸브', '전력']
                    item_name2 = random.choice(asset)
                    cursor.execute(query3, (index + 1, item_name2))

                print('-----------------------')
                print('현재 index', index)


                # 커밋
                # conn.commit()

        except Exception as e:
            print("Error saving to database:", e)
            logging.error(f"Error saving to database: {e}")
        finally:
            conn.close()
        
        print('-----------------------')
        print('현재 index', index)
    except Exception as e:
        print("Error in process_new_data:", e)
        logging.error(f"Error in process_new_data: {e}")


def main():
    try:
        logging.info('프로그램 시작')
        # 주기적으로 process_new_data 함수 실행 (5초마다)
        schedule.every(5).seconds.do(process_new_data)

        # 스케줄러 유지
        while True:
            schedule.run_pending()
            time.sleep(1)

    except Exception as e:
        print("Error fetching data", e)
        logging.error(f"Error in main loop: {e}")

        
if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print(f"Unhandled exception: {e}")
        logging.error(f"Unhandled exception: {e}")


