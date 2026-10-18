from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; CSV=DATA/'student-mat.csv'
URL='https://archive.ics.uci.edu/static/public/320/student+performance.zip'
def load_data():
 DATA.mkdir(exist_ok=True)
 if not CSV.exists():
  print('Downloading UCI Student Performance dataset...')
  try:
   with urlopen(URL,timeout=30) as r: archive=ZipFile(BytesIO(r.read()))
   member=next(n for n in archive.namelist() if n.endswith('student-mat.csv')); CSV.write_bytes(archive.read(member))
  except Exception as e: raise SystemExit(f'Dataset download failed: {e}. Check internet and retry.')
 return pd.read_csv(CSV,sep=';')
def main():
 df=load_data(); X=df.drop(columns=['G3']); y=df['G3']
 cat=X.select_dtypes(include=['object','category']).columns.tolist(); num=X.select_dtypes(exclude=['object','category']).columns.tolist()
 prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num','passthrough',num)])
 model=Pipeline([('prep',prep),('model',RandomForestRegressor(n_estimators=250,random_state=42,min_samples_leaf=2))])
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42); model.fit(Xtr,ytr); pred=model.predict(Xte)
 print(f'Rows: {len(df)} | train: {len(Xtr)} | test: {len(Xte)}')
 print(f'MAE: {mean_absolute_error(yte,pred):.3f}')
 print(f'RMSE: {np.sqrt(mean_squared_error(yte,pred)):.3f}')
 print(f'R²: {r2_score(yte,pred):.3f}')
 print(pd.DataFrame({'actual_G3':yte.iloc[:10].values,'predicted_G3':np.round(pred[:10],2)}).to_string(index=False))
 print('Do not report these metrics on a resume unless you have run and verified this experiment.')
if __name__=='__main__': main()
