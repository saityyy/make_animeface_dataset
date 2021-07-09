# make_animeface_dataset

![diagram](https://user-images.githubusercontent.com/62131533/125000595-af3cb780-e08b-11eb-889a-971528b67dac.png)

# 使い方
1. annotationフォルダを作成し、その直下でgit cloneする。  
`mkdir annotation`  
`cd annotation`  
`git clone https://github.com/saityyy/make_animeface_dataset.git`  
2. 指定のディレクトリ構成になるようにフォルダやファイルを作成する。(temp,image,target.csv)  
3. 取ってきたい画像が入ったフォルダのパスを設定する(check_face.pyのFROMPATH変数)。  
  
  
# 各スクリプトの説明  
- **check_face.py**  
データセットとして使えるかどうかの判定を行う。使えるものはtempフォルダにコピーされる。  

- **fetch_image.py**  
tempフォルダにある画像を適切なファイル名に変更した後、imageフォルダに移す。  

- **annotation.py**  
顔の部分を手動で矩形選択する。矩形のデータはtarget.csvに保存される。  

- **check.py {number}**  
csvファイルの矩形が実際に顔を捉えているか確認する。{number}は画像番号を指定する。  
  
# csvの形式  
  [画像番号,矩形の中心X座標,矩形の中心y座標,矩形のサイズ]
  
