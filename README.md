# Taelim OI

Taeilm 등대공장에서 OI(Operation Inteligence)를 운영함에 있어서 협업할 때 만들었던 모델 정리입니다. 

# 프로젝트 목표

- 스마트 등대공장
현재 조립가공(Assembly) 생산라인에서 생산(OP20, 30), 품질 관련(OP 40) 다양한 데이터를 수집하고 있음

- 다만, 어떤 데이터가 중요한지 사전 탐색이 되어있지 않아서 수집가능한 모든 데이터를 수집 및 관리하고 있음. 이로 인해 데이터 수집, 전송, 관리에 대한 비용이 상당히 많이 발생하고 있음
1차적으로는 생산변수와 품질변수 사이 상관관계 분석(20,30-40)을 통해서 중요 관리 인자를 식별하고, 수집 및 관리 대상을 최소화하고자 함

- 또한, 현재 품질예측 및 관리가 부재하여 중간 품질검사를 수행하고 있음. 따라서, 상관관계 분석 후 품질예측 모델링을 통해서 충분한 신뢰도를 확보하면, 이를 통해 중간 품질검사를 생략하고 예측 값으로 대체하고자 함
  
- 이후 품질 예측 단계에서 품질 불량이 예상될 시 알림 기능 등을 통해서 현장 오퍼레이터의 즉각적인 대응 조치가 가능하게 되기를 희망함

# 프로젝트 과정.

[ EDA ] 
1. 20, 30, 40-S의 공정 내용을 수치형 데이터로 담은 약 20만행의 Excel파일 받음.
2. 20, 30, 40-S 모두 하나의 바코드를 공유함으로써 그것을 기준으로 merge. 20 -> 30 -> 40-S를 가는 하나의 공정이 담긴 CSV파일 생성. 
3. 약 20만개의 CSV데이터를 훈련 및 테스트데이터로 사용. 실제 DB에 존재하는 약 700개의 데이터를 검증용 데이터로 사용하기로 결정.
4. 사전에 받은 개발 사항 관련 문서 및 상관관계, Boxplot등의 분석을 통해 주요 컬럼 분석.
5. 실제 공정 데이터상의 특징 중 하나인 합/불상의 데이터 불균형 발견.
   (합: 195695, 불합: 4425)

[ 데이터 전처리 및 모델링]
1. 데이터를 전처리 하지 않은 Naive한 상태의 데이터 + 튜닝하지 않은 Naive한 모델을 사용했을 때 모델의 Accuracy는 굉장히 높았으나 데이터의 불균형 특성상 나타나는 전형적인 현상인 Recall부분에서 미흡함을 보임. 
Confusion Matrix:
[[26,	916
62,	39064]]

2. 원시적인 데이터 불균형 해결 방법으로 sklearn의 resample기능을 통하여 합격과 불합격 데이터를 4000 vs 4000으로 맞춰서 돌려본 결과 테스트데이터는 Accuracy가 0.8정도 나왔지만 Valid용 데이터로 다른 데이터를 넣어 봤을 때 0.59정도가 나와 과적합됨으로 판단. 모델 폐기.
   
   
3. 데이터 불균형을 해결하기 위한 샘플링 기법 중 하나인 imbalanced-learn에서 제공하는 SMOTETomek(Oversampling+Undersampling) 사용. 합, 불합 각각 191824개의 1:1비율을 가진 약 38만개의 데이터가 생성됨.
   
4. 해당 데이터로 모델링 실시. RandomForest 모델링 결과 0.99의 F1 Score와 함께 모든 부분이 골고루 퍼져 있는 좋은 Confusion Matrix 획득.
Confusion Matrix:
[[37416,  872],
[326, 38116]]

5. Valid용 데이터로 확인했을 때도 0.99의 F1 Score와 좋은 Confusion Matrix로 일반화가 잘 되었다고 판단. 해당 모델 사용하기로 판단.

[ 
12. 기능 구현을 위한 불량 인자 처리를 위해 SHAP Value를 사용하기로 판단. SHAP라이브러리로 모델이 결과값을 내는 데에 있어서 기여한 정도를 칼럼별로 산출.
13. 합격에 대한 모델의 Probability값이 0.9보다 낮을 때(합격률이 90%미만)일 경우에 SHAP Value의 값이 -0.1이하인 칼럼을 불량인자로 판별하기로 결정.
14. 기능 구현의 두번째 과업인 불량 인자에 대한 솔루션 제안의 경우에 Dice_ML에서 제공하는 반사실적 설명(Counterfactual Explantations)을 사용하기로 결정.
15. SHAP Value가 낮은 불량인자에 대한 반사실적 설명으로 불량인자를 어떻게 수정하면 합격률이 올라가는가에 대해 도출. (ex)OP20전장을 -0.03만큼 수정하세요.)
16. DB에 no칼럼을 이용하여 데이터의 실시간 검색 및 모델 예측 기능 구현.












