# Stanford_CS102
> pandas, numpy , simple machine learning

---

### Assignment3

```python
df = pd.read_csv('Players.csv')
# 파일 데이터 읽기
A = df['minutes'].astype('int') < 200
# 조건문으로 뽑아내기
C = df.team.str.contains("ia")
A = df.loc[df['position'] == 'midfielder', ['passes']].mean()
# df.loc를 이용하여 하나의 컬럼으로 만들어준다
new = pd.merge(df,Teams)
# 두 데이터프레임 조인
Result = pd.DataFrame({
    'Southampton': [((tf['embarked'] == 'Southampton') & (tf['survived'] == 'yes')).sum()],
    'Cherbourg': [((tf['embarked'] == 'Cherbourg') & (tf['survived'] == 'yes')).sum()],
    'Queenstown': [((tf['embarked'] == 'Queenstown') & (tf['survived'] == 'yes')).sum()]
})
#데이터프레임 생성
Result = pd.DataFrame(DF)
Result.plot.bar()
# 데이터프레임 막대그래프 표현
ax = fig.add_subplot()
ax.scatter(A['minutes'],A['passes'])
# 데이터프레임 scatter 표현
val = pd.DataFrame(DF['redCards'])
plt.pie(val,
       labels=Teams['team'],
       textprops={'fontsize':20})
#데이터프레임 pie차트 표현

```

### Assignment4

```python
c1 = np.corrcoef(players['passes'],players['minutes'])[1,0]
#correlation 표현
train = players[(players.team =='Greece') | (players.team=='USA') | (players.team=='Portugal')]
# 특정 해당 조건으로만 골라내어 데이터프레임 생성
a,b = np.polyfit(train.minutes,train.passes,1)
# 1은 찾고자 하는 함수의 차수. 데이터에 대한 직선을 찾아준다. 기울기와 절편!!
player.iloc[0].position == 'defender'
# 컬럼명을 인덱스 기준으로 규칙을 찾아낸다
classifier = KNeighborsClassifier(neighbors)
# K -NEAREST NEIGHBORS
dt = DecisionTreeClassifier(random_state=0, min_samples_split=split)
# DECISION TRESS
rf = RandomForestClassifier(random_state=0, min_samples_split=split, n_estimators=trees)
# FOREST of TREES
nb = GaussianNB()
# 가우시안 정규분포
nb.fit(playersTrain[features], playersTrain['position'])
predictions = nb.predict(playersTest[features])
#NAIVE BAYES
#fit 함수로 훈련시킨다
kmeans = cluster.KMeans(10)
kmeans.fit(players[['minutes','passes','shots']])
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(players['minutes'],players['passes'],players['shots'],c=labels)
plt.scatter(centroids[:,0], centroids[:,1],marker='x' ,c='black')
plt.show()
#kmeans를 사용하여 훈련

```

### Assignment5

```python
# 10번 이상 20번 이하의 세 item의 빈번하게 나오는 item-sets의 수
list1 = []
#support = .3
for i1 in items:
    for i2 in items:
        for i3 in items:
            if i1 < i2 and i2 < i3:

                common = len(set(items[i1]) & set(items[i2]) & set(items[i3]))
                
                #print(float(common)/numtrans)
                #print(float(common) / support)
                if float(common) >= 10 and float(common) <= 20:
                    #print (i1, '|', i2, '|', i3)
                    list1.append([i1, '|', i2, '|', i3])

                    
#list1 = set(list1)
print(list1)
```



### Referenced Datasets

 titanic, players , teams , cities, countries