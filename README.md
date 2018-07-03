CheeseBurger.py
    제작 중인 머신러닝 클래스
    우선은 Classifier 기능을 구현해서 Kaggle Titanic 에 써볼 수 있도록.


data/train-sep-for-...
    kaggle 에 올리지 않고 테스트를 하기 위해 train data 를 학습용, 테스트용으로 분리함.
data/train-sep-for-learn.csv
    학습용 749개 row
data/train-sep-for-test.csv
    테스트용 142개 row


## ./exercise
    개발과정 중에 만든 파이썬 문법 및 라이브러리 연습용 파일을 모아둡니다.
    

```
code expression
```

Categorical 한 data 는 scailing 불가, leveling 도 불가.
numerical 한 featrue 는 1~10, 10~20 등으로 묶을 수 있지만, 
categorical 한 data 는 불가능. 
하지만, clustering 으로는 categorical 한 data를 묶을 수 있지 않겠나?

[]clustering 부분 개발.